# Import necessary modules
import hashlib
import numpy as np
import itertools
from multiprocessing import Pool, Manager, set_start_method
import psutil
from tqdm import tqdm
import pickle
import gc, time
from collections import defaultdict

# No more shared memory >:(
set_start_method('spawn', force=True)

cdef class CFRTrainer:
    def __init__(self, int iterations, int num_simulations, int cfr_depth, int num_players, int initial_chips, int small_blind, int big_blind, list bet_sizing, list suits=SUITS, list values=VALUES, int monte_carlo_depth=9999, int prune_depth = 9999, double prune_probability = 1e-8, external_manager=None):
        self.iterations = iterations
        self.num_simulations = num_simulations

        self.cfr_depth = cfr_depth

        self.prune_depth = prune_depth
        self.prune_probability_threshold = prune_probability

        self.num_players = num_players
        self.initial_chips = initial_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.bet_sizing = bet_sizing

        self.suits = suits
        self.values = values

        self.monte_carlo_depth = monte_carlo_depth
    

    cpdef train(self, list positions_to_solve = [], list hands = [], bint save_pickle=False, bint mem_efficient=False):
        FUNCTION_START_TIME = time.time()

        cards = [r + s for r in reversed(self.values) for s in self.suits]

        hands = hands if hands else list(sorted(itertools.combinations(cards, 2)))

        hand_mapping = {}
        for hand in hands:
            card1, card2 = hand
            abstracted_hand = abstract_hand(card_str_to_int(card1), card_str_to_int(card2))
            hand_mapping[hand] = abstracted_hand

        # TODO generalize this away from being preflop abstractions only.
        hands_reduced = []
        __ = {}
        for hand in hands:
            if hand_mapping[hand] in __:
                continue
            
            __[hand_mapping[hand]] = 1
            hands_reduced.append(hand)


        ## TODO: Figure out how to use manager.dict() with ExternalManager
        external_manager = ExternalManager('dat/pickles/regret_sum.pkl', 'dat/pickles/strategy_sum.pkl')

        # NOTE: Migrate away from passing directories
        # TODO: Pass positions_to_solve to ExternalManager; make mappings from positions to their root strategy nodes.

        local_hand_strategy_aggregate = []

        for fast_forward_actions in positions_to_solve:
            external_manager, hand_strategy_aggregate = self.parallel_train(hands_reduced, hand_mapping, fast_forward_actions, external_manager, save_pickle, mem_efficient)

            local_hand_strategy_aggregate.extend(hand_strategy_aggregate)

            if save_pickle:
                external_manager.save('dat/pickles/regret_sum.pkl', 'dat/pickles/strategy_sum.pkl')

        print(f'Time taken: {time.time() - FUNCTION_START_TIME}')

        return list(local_hand_strategy_aggregate), external_manager

    def parallel_train(self, hands, hand_mapping, fast_forward_actions, external_manager, save_pickle, mem_efficient, batch_size=psutil.cpu_count(logical=True) * 4):

        ### Managed Objects
        manager = Manager()
        train_regret = manager.dict()
        train_strategy = manager.dict()
        hand_strategy_aggregate = manager.list()
        calculated = manager.dict()


        def process_batch(batch_hands):
            hands = [(hand, hand_mapping, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions, external_manager) for hand in batch_hands]
            with Pool(processes=psutil.cpu_count(logical=True)) as pool:
                try:
                    pool.starmap(self.process_hand_wrapper, hands)
                finally:
                    pool.close()
                    pool.join()

        num_batches = (len(hands) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches)):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(hands))
            batch_hands = hands[start_index:end_index]

            process_batch(batch_hands)

            if not mem_efficient:
                # Migrate local solutions to global solutions
                external_manager.get_regret_sum().update(dict(train_regret))
                external_manager.get_strategy_sum().update(dict(train_strategy))
            
            if save_pickle:
                external_manager.save('dat/pickles/regret_sum.pkl', 'dat/pickles/strategy_sum.pkl')
                
            # Clear local solutions for next run
            train_regret.clear()
            train_strategy.clear()
            gc.collect()

        return external_manager, hand_strategy_aggregate


    def process_hand_wrapper(self, hand, hand_mapping, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions, external_manager):
        self.process_hand(hand, hand_mapping, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions, external_manager)


    def process_hand(self, hand, hand_mapping, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions, external_manager):
        if calculated.get(hand_mapping[hand], False):
            return
        calculated[hand_mapping[hand]] = True

        cdef GameState game_state = GameState([Player(self.initial_chips, self.bet_sizing, False) for _ in range(self.num_players)], self.small_blind, self.big_blind, self.num_simulations, True, self.suits, self.values) 
        # Fastforward to current node for debug purposes.
        self.fast_forward_gamestate(hand, game_state, fast_forward_actions, external_manager)
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print(f'TRAINING: ({game_state.get_current_player().position}) --- {hand}')
        print(f'Regret Complexity: {len(external_manager.get_regret_sum())}')
        print(f'Strategy Complexity: {len(external_manager.get_strategy_sum())}')
        print()
        player = game_state.get_current_player()
        strategy = self.get_average_strategy(player, game_state, external_manager)
        player_hash = player.hash(game_state)
        print(player_hash)
        print(f"Init Regret Sum: {external_manager.get_regret_sum().get(player_hash, defaultdict(self.default_double))}")
        print(f"Init Strategy Sum: {external_manager.get_strategy_sum().get(player_hash, defaultdict(self.default_double))}")
        print(f"Init Average Strategy For Hand {hand_mapping[hand]}: {strategy}")
        
        probs = np.ones(self.num_players, dtype=np.float64)

        for iter_num in tqdm(range(self.iterations)):
            epsilon = 0.00 * (0.9999 ** iter_num)

            probs.fill(1)
            self.fast_forward_gamestate(hand, game_state, fast_forward_actions, external_manager)

            self.cfr_traverse(game_state, probs, 0, self.cfr_depth, epsilon, external_manager)


        
        strategy = self.get_average_strategy(player, game_state, external_manager)
        print(player_hash)
        print(f"Post Regret Sum: {external_manager.get_regret_sum()[player_hash]}")
        print(f"Post Strategy Sum: {external_manager.get_strategy_sum()[player_hash]}")
        print(f"Post Average Strategy For Hand {hand_mapping[hand]}: {strategy}")

        ### Prune local regret and strategy sums to save memory at the global level.
        external_manager.get_regret_sum().prune()
        external_manager.get_strategy_sum().prune()

        ### Merge Local Results with Global Results
        dynamic_merge_dicts(external_manager.get_regret_sum(), train_regret)
        dynamic_merge_dicts(external_manager.get_strategy_sum(), train_strategy)

        hand_strategy_aggregate.append((player.position, fast_forward_actions, hand_mapping[hand], strategy))


    cdef GameState fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions, ExternalManager external_manager):
        ### NOTE: This function is used to walk a gamestate to a possible point in the current GTO gametree.
        ### TODO: Incorporate construction of PROBS array.
        game_state.setup_preflop(hand)
        for action in fast_forward_actions:
            strategy = self.get_average_strategy(game_state.get_current_player(), game_state, external_manager)

            player_hash = game_state.get_current_player().hash(game_state)

            # Define an arbitrary 'optimal threshold' that ensures the selected strategy isn't completely arbitrary given the action space
            optim_thresh = 1/max(len(strategy), 1)**2
            
            # if the action is not in the current blueprint, that's ok. Just means we didnt visit this node during training.
            #  Especially if you're training in realtime and the blueprint hasn't been updated yet.
            if action not in strategy:
                pass # No-ops are cool if the alternative is making
                     # excessively long conditionals like below worse.
            
            elif strategy[action] < optim_thresh:
                # The selected action is not optimal. Construct new gamestate
                self.fast_forward_gamestate(hand, game_state, fast_forward_actions, external_manager)
                break


            # action is blueprint compliant
            game_state.step(action)

        if hand:
            game_state.update_current_hand(hand)

    cpdef default_double(self):
        ### 
        return 0.0

    cdef progress_gamestate_to_showdown(self, GameState game_state, float epsilon = 1):
        game_state.progress_to_showdown()

    cdef double[:] cfr_traverse(self, GameState game_state, double[:] probs, int depth, int max_depth, float epsilon, ExternalManager external_manager):
        cdef int cur_player_index = game_state.player_index
        cdef Player cur_player = game_state.get_current_player()
        cdef object action = ()

        cdef int num_players = len(game_state.players)
        cdef double[:] node_util = cython.view.array(shape=(num_players,), itemsize=sizeof(double), format="d")
        cdef dict[double[:]] util
        cdef double[:] new_probs = cython.view.array(shape=(num_players,), itemsize=sizeof(double), format="d")
        cdef dict strategy
        cdef int action_index
        cdef float rand_value, uniform_prob
        cdef double opp_contribution
        cdef double regret
        cdef int monte_carlo = depth >= self.monte_carlo_depth
        cdef int prune = depth >= self.prune_depth
        cdef list available_actions = game_state.get_current_player().get_available_actions(game_state)

        if game_state.is_terminal_river() or depth >= max_depth or probs[cur_player_index] < self.prune_probability_threshold:
            self.progress_gamestate_to_showdown(game_state)
            return self.calculate_utilities(game_state, game_state.winner_index)

        player_hash = cur_player.hash(game_state)
        external_manager.get_regret_sum()[player_hash] = (external_manager.get_regret_sum().get(player_hash, defaultdict(self.default_double)), prune)

        util = {action: np.zeros(num_players, dtype=np.float64) for action in available_actions}
        strategy = self.get_strategy(available_actions, probs, game_state, cur_player, prune, external_manager)


        if monte_carlo and available_actions:
            epsilon_calc = epsilon
            rand_value = np.random.rand()

            if rand_value < epsilon_calc:
                # Randomly explore
                uniform_prob = 1.0 / len(available_actions)
                for action in available_actions:
                    strategy[action] = uniform_prob

            available_actions = [select_action(strategy)]

        for action in available_actions:
            new_game_state = game_state.clone()
            if new_game_state.step(action):
                probs[:] = 1.0
            new_probs[:] = probs
            new_probs[cur_player_index] *= strategy[action]
            util[action] = self.cfr_traverse(new_game_state, new_probs, depth + 1, max_depth, epsilon, external_manager)

        for i in range(num_players):
            node_util[i] = sum(strategy[action] * util[action][i] for action in available_actions)

        for action in available_actions:
            regret = util[action][cur_player_index] - node_util[cur_player_index]
            opp_contribution = 1.0
            for i in range(num_players):
                if i != cur_player_index:
                    opp_contribution *= probs[i]
            external_manager.get_regret_sum()[player_hash][action] += opp_contribution * regret

        return node_util

    cdef double[:] calculate_utilities(self, GameState game_state, int winner_index):
        cdef int num_players = len(game_state.players)
        cdef double[:] utilities = np.zeros(num_players, dtype=np.float64)

        rake = game_state.pot * .03
        rake = rake if rake < 4 else 4
        pot = game_state.pot - rake
        for i, p in enumerate(game_state.players):
            if i == winner_index:
                                                                # NOTE: the idea is to punish the CFR more for participating in multiway pots.
                utilities[i] = (pot - p.tot_contributed_to_pot)#/(game_state.active_players() if game_state.active_players() > 2 else 1)
            else:
                utilities[i] = -(p.tot_contributed_to_pot)

        return utilities

    cdef dict get_strategy(self, list available_actions, double[:] probs, GameState game_state, Player player, bint prune, ExternalManager external_manager):
        current_player = game_state.player_index
        player_hash = player.hash(game_state)
        external_manager.get_strategy_sum()[player_hash] = (external_manager.get_strategy_sum().get(player_hash, defaultdict(self.default_double)), prune)

        if not available_actions:
            return {}

        strategy = {}
        normalization_sum = 0.0
        regrets = [external_manager.get_regret_sum()[player_hash][action] for action in available_actions]
        regrets_norm = [max(x, 0) for x in regrets]

        for action, regret in zip(available_actions, regrets_norm):
            strategy[action] = regret
            normalization_sum += regret

        if normalization_sum > 0:
            for action in strategy:
                strategy[action] /= normalization_sum
                external_manager.get_strategy_sum()[player_hash][action] += probs[current_player] * strategy[action]
        elif sum(regrets) == 0: # new node
            uniform_prob = 1 / len(available_actions)
            for action in strategy:
                strategy[action] = uniform_prob
                external_manager.get_strategy_sum()[player_hash][action] += probs[current_player] * uniform_prob
        elif all([x <= 0 for x in regrets]): # only negative regret
            if ('fold', 0) in strategy:
                for action in strategy:
                    strategy[action] = 1.0 if action[0] == 'fold' else 0.0
            else:
                for action in strategy:
                    strategy[action] = 1.0 if action[0] == 'call' else 0.0
        else:
            raise 'Unexpected regret combination. Please see "get_strategy'



        return strategy


    cpdef dict get_average_strategy(self, Player player, GameState game_state, ExternalManager external_manager):
        average_strategy = {}
        current_player = game_state.player_index
        game_state_hash = player.hash(game_state)
        available_actions = player.get_available_actions(game_state)
        cur_gamestate_strategy = external_manager.get_strategy_sum().get(game_state_hash, defaultdict(self.default_double))

        normalization_sum = 0.0
        regrets = [cur_gamestate_strategy[action] for action in available_actions]
        regrets_norm = [max(x, 0) for x in regrets]

        for action, regret in zip(cur_gamestate_strategy, regrets_norm):
            average_strategy[action] = regret
            normalization_sum += regret

        if normalization_sum > 0:
            for action in average_strategy:
                average_strategy[action] /= normalization_sum
        elif sum(regrets) == 0:  # new node
            uniform_prob = 1 / len(available_actions)
            for action in average_strategy:
                average_strategy[action] = uniform_prob
        elif all([x <= 0 for x in regrets]):  # only negative regret
            if ('fold', 0) in average_strategy:
                for action in average_strategy:
                    average_strategy[action] = 1 if action[0] == 'fold' else 0
            else:
                for action in average_strategy:
                    average_strategy[action] = 1 if action[0] == 'call' else 0
        else:
            raise 'Unexpected regret combination. Please see "get_average_strategy"'

        return average_strategy

