# cython: language_level=3

import hashlib
import numpy as np
cimport numpy as np
import random
import itertools
from multiprocessing import Pool, Manager, set_start_method
import psutil
from tqdm import tqdm
import pickle
import gc, time
from collections import defaultdict
from cython.view cimport array

# No more shared memory >:(
set_start_method('spawn', force=True)

cdef class CFRTrainer:
    def __init__(self, int iterations, int cfr_depth, int num_players, int initial_chips, int small_blind, int big_blind, list bet_sizing, list suits=SUITS, list values=VALUES, int monte_carlo_depth=9999, int prune_depth = 9999, double prune_probability = 1e-8, local_manager=None):
        self.iterations = iterations

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
    
    cpdef get_average_strategy_dump(self, fast_forward_actions, local_manager):
        cdef GameState game_state = GameState([Player(self.initial_chips, self.bet_sizing, False) for _ in range(self.num_players)], self.small_blind, self.big_blind, True, self.suits, self.values)
        aggregate_hand_dump = []
        hands = self.generate_hands()
        hands = [(card_str_to_int(hand[0]), card_str_to_int(hand[1])) for hand in hands]
        for hand in tqdm(hands):

            ffw_probs = self.fast_forward_gamestate(hand, game_state, fast_forward_actions, local_manager)
            
            player = game_state.get_current_player()
            strategy = self.get_average_strategy(player, game_state, local_manager)
            cur_hand_dump = (player.position, fast_forward_actions, abstract_hand(hand[0], hand[1]), strategy, ffw_probs[game_state.player_index])
            aggregate_hand_dump.append(cur_hand_dump)

        return aggregate_hand_dump
        
    cpdef generate_hands(self):
        cards = [r + s for r in reversed(self.values) for s in self.suits]

        hands = list(sorted(itertools.combinations(cards, 2)))
        
        return hands
        
    cpdef train(self, local_manager, list positions_to_solve = [], list hands = [], bint save_pickle=False):
        FUNCTION_START_TIME = time.time()
        
        if not hands:
            hands = self.generate_hands()

        # Convert string hands to unsigned long long hands.
        hands = [(card_str_to_int(hand[0]), card_str_to_int(hand[1])) for hand in hands]        

        # Shuffle the list with a specific seed for reproducibility
        rng = random.Random(42)  # Use any seed value you prefer
        rng.shuffle(hands)

        for i, fast_forward_action in enumerate(positions_to_solve):
            
            local_manager = self.parallel_train(hands, fast_forward_action, local_manager, save_pickle)

            if save_pickle:
                local_manager.save()  

        print(f'Time taken: {time.time() - FUNCTION_START_TIME}')

        return local_manager

    def parallel_train(self, hands, fast_forward_actions, local_manager, save_pickle):#(psutil.cpu_count(logical=True) -1) * 2

        ### Managed Objects
        manager = Manager()
        regret_global_accumulator = manager.dict()
        strategy_global_accumulator = manager.dict()
        calculated = manager.dict()


        def process_batch(batch_hands):                                                                        # NOTE: Since we _spawn_ each pooled process, local_manager gets copied to each child process. If the blueprint gets too large, we make local manager read only and global; then only update a local copy of the explorable nodes.
            hands = [(hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager) for hand in batch_hands]
            with Pool(processes = psutil.cpu_count(logical=True) - 1) as pool: #psutil.cpu_count(logical=True) - 1
                try:
                    pool.starmap(self.process_hand_wrapper, hands)
                except Exception as e:
                    pool.terminate()
                    raise
                finally:
                    pool.close()
                    pool.join()

        
        
        batch_size = (psutil.cpu_count(logical=True) - 1)

        num_batches = (len(hands) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches)):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(hands))
            batch_hands = hands[start_index:end_index]

            process_batch(batch_hands)

            # Migrate local solutions to global solutions
            local_manager.get_regret_sum().update(dict(regret_global_accumulator))
            local_manager.get_strategy_sum().update(dict(strategy_global_accumulator))
            
            if save_pickle:
                local_manager.save()  

#############
            # print(len(local_manager.get_regret_sum().table))
            # print(len(local_manager.get_strategy_sum().table))
#############


            # NOTE: This is an approximate calculation. real calc is something like:
            #       for each hand considered, all villian responses will not have this hand.
            #       This is card removal. Just do the math they say.. but i'm bad at combinatorics i say.
            #   card_removal_effect = | union_of_all_action_spaces - union_of_all_but_one_hand_action_space |
            #   leading to the complexity calculation :
            #   complexity_calculation = (len(global_accumulator) - (card_removal_effect * batch_hands))/batch_hands
            #  The complexity_calculation would give insight into how many nodes are explored and updated for a given gamestate.
            
            regret_complexity_calculation = (len(regret_global_accumulator))/len(batch_hands)
            strategy_complexity_calculation = (len(strategy_global_accumulator))/len(batch_hands)

#############
            # print(f'Batch size: {len(batch_hands)}')
            print(f'Update Complexity: {regret_complexity_calculation if self.cfr_depth > 1 else 1}')
            print(f'Update Complexity: {strategy_complexity_calculation if self.cfr_depth > 1 else 1}')
#############

            # Clear local solutions for next run
            regret_global_accumulator.clear()
            strategy_global_accumulator.clear()
            gc.collect()

        return local_manager


    def process_hand_wrapper(self, hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager):
        self.process_hand(hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager)


    def process_hand(self, hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager):
        cdef GameState game_state = GameState([Player(self.initial_chips, self.bet_sizing, False) for _ in range(self.num_players)], self.small_blind, self.big_blind, True, self.suits, self.values) 
        # Fastforward to current node for debug purposes.
        avg_probs = np.zeros(self.num_players, dtype=np.float64)
#############
        # print('fastforwarding')
        ffw_probs = self.fast_forward_gamestate(hand, game_state, fast_forward_actions, local_manager)

        if ffw_probs == 'ERROR':
            print('Provided hand is incompatible with Fast Forward Actions')
            return

        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print(f'TRAINING: ({game_state.get_current_player().position}) --- {abstract_hand(hand[0], hand[1])}')
        print(fast_forward_actions)
        print(f'Regret Complexity: {len(local_manager.get_regret_sum())}')
        print(f'Strategy Complexity: {len(local_manager.get_strategy_sum())}')
        # print()
#############

        player = game_state.get_current_player()
        strategy = self.get_average_strategy(player, game_state, local_manager)
        player_hash = player.hash(game_state)

#############
        print(player_hash)
        print(f"Init Regret Sum: {local_manager.get_regret_sum().get(player_hash, defaultdict(self.default_double))}")
        print(f"Init Strategy Sum: {local_manager.get_strategy_sum().get(player_hash, defaultdict(self.default_double))}")
        print(f"Init Average Strategy For Hand {abstract_hand(hand[0], hand[1])}: {strategy}")
#############
        total_fails = 0
        for iter_num in tqdm(range(self.iterations)):
            epsilon = 0.00 * (0.9999 ** iter_num)

            ffw_probs = self.fast_forward_gamestate(hand, game_state, fast_forward_actions, local_manager)

            avg_probs += ffw_probs
            
            if ffw_probs[game_state.player_index] < 1e-6:
                total_fails += 1
                # continue

            self.cfr_traverse(game_state.clone(), ffw_probs, 0, self.cfr_depth, epsilon, local_manager)

        if self.iterations * .9 < total_fails:
            print(f'Node unlikely. Current hand: {abstract_hand(hand[0], hand[1])}')
        
        ffw_probs = avg_probs / (self.iterations)
        print(total_fails)

        player = game_state.get_current_player()
        player_hash = player.hash(game_state)


        
#############
        ## NOTE: Output traversal strategy for current node.
        strategy = self.get_strategy(player.get_available_actions(game_state), ffw_probs, game_state, player, local_manager)
        regret = local_manager.get_regret_sum()[player_hash]
        print(f"Node Reach Probability: {list(ffw_probs)[game_state.player_index]}")
        print(player_hash)

        print(f"Post Strategy For Hand {abstract_hand(hand[0], hand[1])}: {strategy}")
        print(f"Post Regret Sums For Hand {abstract_hand(hand[0], hand[1])}: {regret}")
#############
        ## NOTE: Average strategy is for the user
        strategy = self.get_average_strategy(player, game_state, local_manager)
        print(f"Post Average Strategy For Hand {abstract_hand(hand[0], hand[1])}: {strategy}")
#############

        ### Prune local regret and strategy sums to save memory at the global level.
        local_manager.get_regret_sum().prune()
        local_manager.get_strategy_sum().prune()
    

        ### Merge Local Results with Global Accumulator
        dynamic_merge_dicts(local_manager.get_regret_sum(), regret_global_accumulator)
        dynamic_merge_dicts(local_manager.get_strategy_sum(), strategy_global_accumulator)

    cdef fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions, LocalManager local_manager, int attempts = 0):
        cdef dict strategy
        cdef list ignore_cards
        cdef GameState cloned_game_state
        cdef double[:] new_probs
        cdef double total
        cdef object action
        cdef Player current_player
        
        # Accumulate cards that need to be handled specially
        ignore_cards = [action[1] for action in fast_forward_actions if action[0] == 'PUBLIC']
        ignore_cards.extend(hand)

        if len(set(ignore_cards)) != len(ignore_cards):
            return 'ERROR'

        # Set up preflop - not dealing any card that appears in the list of ignore_cards
        game_state.setup_preflop(ignore_cards)

        # Create game_state representation to set player hand
        cloned_game_state = game_state.clone()
        for action in fast_forward_actions:
            if action[0] != 'PUBLIC':
                cloned_game_state.step(action)

        # Save the current player index
        player_index_tmp = game_state.player_index

        # Update game_state with the cloned game state
        game_state.player_index = cloned_game_state.player_index

        # Update player hand with the provided hand
        game_state.update_current_hand(hand)

        # Restore the original player index
        game_state.player_index = player_index_tmp

        # Set probability space for CFR traversal
        new_probs = np.ones(self.num_players, dtype=np.float64)

        for action in fast_forward_actions:
            if action[0] == 'PUBLIC':
                # Add custom card to the board and update action space
                game_state.board |= action[1]
                game_state.action_space[game_state.cur_round_index].insert(0, ('PUBLIC', action))
                continue

            # Get the CFR strategy
            current_player = game_state.get_current_player()
            strategy = self.get_strategy(current_player.get_available_actions(game_state), new_probs, game_state, current_player, local_manager)

            # Update probabilities
            new_probs[game_state.player_index] *= strategy[action]

            if game_state.step(action) and game_state.cur_round_index < 4:
                # Handle the end of the round
                ignore_cards = [new_action for new_action in game_state.action_space[game_state.cur_round_index] if new_action[0] == 'PUBLIC']

                for new_action in ignore_cards:
                    game_state.board -= new_action[1][1]
                    game_state.deck.add(new_action[1][1])
                    game_state.action_space[game_state.cur_round_index].remove(new_action)

                # Normalize probabilities
                total = np.sum(new_probs)
                if total > 0:
                    for i in range(self.num_players):
                        new_probs[i] /= total
                else:
                    for i in range(self.num_players):
                        new_probs[i] = 1

        return new_probs



    cpdef double default_double(self):
        ### 
        return 0.0

    cdef progress_gamestate_to_showdown(self, GameState game_state, float epsilon = 1):
        ###
        ### wrapper; i considered performing monte-carlo sampling for a number of iterations
        ###    But i figured that would be too much too soon.
        ### TODO Investigate the feasibility of montecarlo terminal sampling
        ###
        game_state.progress_to_showdown()

    cdef double[:] cfr_traverse(self, GameState game_state, double[:] probs, int depth, int max_depth, float epsilon, LocalManager local_manager):
        cdef int cur_player_index = game_state.player_index
        cdef Player cur_player = game_state.get_current_player()
        cdef object action = ()
        
        cdef int num_players = len(game_state.players)
        cdef double[:] node_util = array(shape=(num_players,), itemsize=sizeof(double), format="d")
        cdef dict[double[:]] util
        cdef double[:] new_probs = array(shape=(num_players,), itemsize=sizeof(double), format="d")
        cdef dict strategy
        cdef int action_index
        cdef double opp_contribution, regret
        cdef list available_actions = game_state.get_current_player().get_available_actions(game_state)

        cdef bint monte_carlo = depth >= self.monte_carlo_depth
        cdef bint merge_criteria = depth == 0
        cdef bint prune_criteria = depth >= self.prune_depth
        cdef bint depth_criteria = not cur_player.folded
        
        cdef object player_hash = cur_player.hash(game_state)

        regret_sum = local_manager.get_regret_sum().get_set(player_hash, defaultdict(self.default_double), prune_criteria, merge_criteria)
        strategy_sum = local_manager.get_strategy_sum().get_set(player_hash, defaultdict(self.default_double), prune_criteria, merge_criteria)

        if game_state.is_terminal_river() or depth >= max_depth or probs[cur_player_index] < self.prune_probability_threshold:
            self.progress_gamestate_to_showdown(game_state)
            return self.calculate_utilities(game_state, game_state.winner_index)

        util = {action: np.zeros(num_players, dtype=np.float64) for action in available_actions}
        strategy = self.get_strategy(available_actions, probs, game_state, cur_player, local_manager)

        for action in available_actions:
            new_game_state = game_state.clone()
            new_probs[:] = probs
            if new_game_state.step(action):
                total = np.sum(new_probs)
                new_probs = new_probs / total if total > 0 else new_probs.fill(1)
            
            new_probs[cur_player_index] *= strategy[action]
            util[action] = self.cfr_traverse(new_game_state, new_probs, depth + depth_criteria, max_depth, epsilon, local_manager)

        for i in range(num_players):
            node_util[i] = sum(strategy[action] * util[action][i] for action in available_actions)

        for action in available_actions:
            regret = util[action][cur_player_index] - node_util[cur_player_index]
            opp_contribution = np.prod([probs[i] for i in range(num_players) if i != cur_player_index])
            regret_sum[action] += opp_contribution * regret

        return node_util

    cdef double[:] calculate_utilities(self, GameState game_state, int winner_index):
        cdef int num_players
        cdef double[:] utilities
        cdef double rake
        cdef double pot
        cdef int i
        cdef Player p

        num_players = len(game_state.players)
        utilities = np.zeros(num_players, dtype=np.float64)

        rake = game_state.pot * .05
        rake = rake // 1 if rake < 6 else 6
        pot = game_state.pot - rake

        for i, p in enumerate(game_state.players):
            if i == winner_index:
                # NOTE: the idea is to punish the CFR more for participating in multiway pots.
                utilities[i] = (pot - p.tot_contributed_to_pot)  # /(game_state.active_players() if game_state.active_players() > 2 else 1)
            else:
                utilities[i] = -(p.tot_contributed_to_pot)

        return utilities


    cdef dict get_strategy(self, list available_actions, double[:] probs, GameState game_state, Player player, LocalManager local_manager):
        cdef int current_player
        cdef object player_hash
        cdef dict strategy
        cdef double normalization_sum, uniform_prob, regret, min_regret, total_positive_regrets
        cdef list regrets, regrets_norm
        cdef object action

        current_player = game_state.player_index
        player_hash = player.hash(game_state)

        if not available_actions:
            return {}

        regret_sum = local_manager.get_regret_sum().get_set(player_hash, defaultdict(self.default_double))
        strategy_sum = local_manager.get_strategy_sum().get_set(player_hash, defaultdict(self.default_double))

        strategy = {}
        normalization_sum = 0.0
        regrets = [regret_sum[action] for action in available_actions]
        regrets_norm = [max(x, 0) for x in regrets]

        normalization_sum = sum(regrets_norm)

        if normalization_sum > 0:
            for action, regret in zip(available_actions, regrets_norm):
                strategy[action] = regret / normalization_sum
                strategy_sum[action] += probs[current_player] * strategy[action]
        elif all(x == 0 for x in regrets):  # new node
            uniform_prob = 1.0 / len(available_actions)
            for action in available_actions:
                strategy[action] = uniform_prob
                strategy_sum[action] += probs[current_player] * uniform_prob
        elif all(x <= 0 for x in regrets):  # only negative regret
            min_regret = min(regrets)
            positive_regrets = [regret - min_regret for regret in regrets]
            total_positive_regrets = sum(positive_regrets)
            for action, positive_regret in zip(available_actions, positive_regrets):
                strategy[action] = positive_regret / total_positive_regrets
                strategy_sum[action] += probs[current_player] * strategy[action]
        else:
            raise ValueError('Unexpected regret combination. Please see "get_strategy"')

        return strategy

    cpdef dict get_average_strategy(self, Player player, GameState game_state, LocalManager local_manager):
        cdef object average_strategy
        cdef int current_player
        cdef object game_state_hash
        cdef list available_actions
        cdef object cur_gamestate_strategy
        cdef double normalization_sum
        cdef list regrets
        cdef list regrets_norm
        cdef double uniform_prob, regret
        cdef object action

        average_strategy = {}
        current_player = game_state.player_index
        game_state_hash = player.hash(game_state)
        available_actions = player.get_available_actions(game_state)
        cur_gamestate_strategy = local_manager.get_strategy_sum().get(game_state_hash, defaultdict(self.default_double))

        normalization_sum = 0.0
        regrets = [cur_gamestate_strategy[action] for action in available_actions]
        regrets_norm = [max(x, 0) for x in regrets]

        for action, regret in zip(cur_gamestate_strategy, regrets_norm):
            average_strategy[action] = regret
            normalization_sum += regret

        if normalization_sum > 0:
            for action in average_strategy:
                average_strategy[action] /= normalization_sum
        elif all([x == 0 for x in regrets]):  # new node
            uniform_prob = 1 / len(available_actions)
            for action in average_strategy:
                average_strategy[action] = uniform_prob
        elif all([x <= 0 for x in regrets]):  # only negative regret
            # Convert negative regrets to positive values by adding the minimum regret value to each
            min_regret = min(regrets)
            positive_regrets = [regret - min_regret for regret in regrets]
            total_positive_regrets = sum(positive_regrets)
            for action, positive_regret in zip(average_strategy.keys(), positive_regrets):
                average_strategy[action] = positive_regret / total_positive_regrets

        # else:
        #     raise 'Unexpected regret combination. Please see "get_average_strategy"'

        return average_strategy
