# Import necessary modules
import hashlib
import numpy as np
import itertools
from multiprocessing import Pool, Manager
import psutil
from tqdm import tqdm
import pickle
import gc, time
from collections import defaultdict

cdef class CFRTrainer:
    def __init__(self, int iterations, int realtime_iterations, int num_simulations, int cfr_depth, int cfr_realtime_depth, int num_players, int initial_chips, int small_blind, int big_blind, list bet_sizing, list suits=SUITS, list values=VALUES, int monte_carlo_depth=9999, int prune_depth = 9999, double prune_probability = 1e-8):
        self.iterations = iterations
        self.realtime_iterations = realtime_iterations
        self.num_simulations = num_simulations
        self.cfr_realtime_depth = cfr_realtime_depth
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

        # Use HashTable for regret_sum and strategy_sum
        self.regret_sum = HashTable()
        self.strategy_sum = HashTable()

    cpdef train(self, list positions_to_solve = []):
        cdef double[:] probs

        # Generate all possible cards
        cards = [r + s for r in reversed(self.values) for s in self.suits]

        # Generate all possible hands (for testing, we can sample this to focus on the first n to maximize iteration potential)
        hands = list(sorted(itertools.combinations(cards, 2)))

        # Create a mapping from hands tuple to abstracted representation
        hand_mapping = {}
        for hand in hands:
            card1, card2 = hand
            rank1, suit1 = card1[:-1], card1[-1]
            rank2, suit2 = card2[:-1], card2[-1]

            if suit1 == suit2:
                abstracted_hand = rank1 + rank2 + 's'
            else:
                abstracted_hand = rank1 + rank2 + 'o'

            hand_mapping[hand] = abstracted_hand

        # Define a new game state for training
        players = [AIPlayer(self.initial_chips, self.bet_sizing, self) for _ in range(self.num_players)]
        
        # Define initial gamestate
        game_state = GameState(players, self.small_blind, self.big_blind, self.num_simulations, True, self.suits, self.values)

        # Reduce the hands to uniquely abstractable hands
        hands_reduced = []
        __ = {}
        for hand in hands:
            if hand_mapping[hand] in __:
                continue
            
            __[hand_mapping[hand]] = 1
            hands_reduced.append(hand)

        local_hand_strategy_aggregate = []

        for fast_forward_actions in positions_to_solve:
            # Call the parallel training function
            train_regret, train_strategy, hand_strategy_aggregate = self.parallel_train(hands_reduced, hand_mapping, players, game_state, fast_forward_actions)

            # Set the global regret_sum and strategy_sum to the training values
            self.regret_sum.update(dict(train_regret))
            self.strategy_sum.update(dict(train_strategy))
            local_hand_strategy_aggregate.extend(hand_strategy_aggregate)

        # Pickle and save regret and strategy sums
        with open('../dat/pickles/regret_sum.pkl', 'wb') as f:
            pickle.dump(self.regret_sum, f)

        with open('../dat/pickles/strategy_sum.pkl', 'wb') as f:
            pickle.dump(self.strategy_sum, f)
        
        return list(local_hand_strategy_aggregate)

    def process_hand(self, hand, hand_mapping, players, game_state, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions):
        """
        Solve for the current player in the gamestate for the given hand.
        """

        if calculated.get(hand_mapping[hand], False):
            return
        calculated[hand_mapping[hand]] = True

        game_state = game_state.clone()
        
        print(f'Current Hand: {hand}')
        print(game_state)
        

        local_regret_sum = HashTable()
        local_strategy_sum = HashTable()
        local_hand_strategy_aggregate = []

        # Initialize probabilities
        probs = np.ones(self.num_players, dtype=np.float64)

        for iter_num in tqdm(range(self.iterations)):
            epsilon = 0.00 * (0.9999 ** iter_num)  # Update epsilon based on iteration

            probs.fill(1)

            # Set preflop so opponent sees new hand each iteration. 
            game_state.setup_preflop(hand)

            # NOTE We want to fast forward to a valid GTO state.. TODO: Implement this.
            self.fast_forward_gamestate(hand, game_state, fast_forward_actions)

            # Perform CFR traversal for the current hand
            self.cfr_traverse(game_state.clone(), probs, 0, self.cfr_depth, epsilon)

            # If using monte carlo - remove all monte carlo samples.
            self.strategy_sum.prune()
            self.regret_sum.prune()

        # Extract current strategy.
        player_idx = game_state.player_index
        player = players[player_idx]
        strategy = self.get_average_strategy(player, game_state)
        player_hash = player.hash(game_state)
        print(player_hash)
        print(f"Regret sum: {self.regret_sum[player_hash]}")
        print(f"Strategy sum: {self.strategy_sum[player_hash]}")
        print(f"Average strategy for hand {hand_mapping[hand]}: {strategy}")

        # Prune none-starting nodes.
        local_regret_sum[player_hash] = (self.regret_sum[player_hash], False)
        local_strategy_sum[player_hash] = (self.strategy_sum[player_hash], False)

        self.regret_sum.clear()
        self.strategy_sum.clear()

        local_hand_strategy_aggregate.append((player.position, fast_forward_actions, hand_mapping[hand], strategy))

        # Perform batch update to shared dictionaries
        train_regret.update(local_regret_sum)
        train_strategy.update(local_strategy_sum)
        hand_strategy_aggregate.extend(local_hand_strategy_aggregate)

    cdef fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions):
        for action in fast_forward_actions:
            player_idx = game_state.player_index
            strategy = self.get_average_strategy(game_state.players[player_idx], game_state)

            player_hash = game_state.players[player_idx].hash(game_state)

            # NOTE: Verify the action aligns with the current strategy.
            if strategy[action] < .10 or player_hash not in self.strategy_sum:
                ## Current game_node is non-gto; reset and try again.
                # TODO: Odds of never breaking out of this non zero?
                game_state.setup_preflop(hand)
                self.fast_forward_gamestate(hand, game_state, fast_forward_actions)
                break

            # Assuming the action aligns with the current strategy, perform the action.
            if game_state.handle_action(action):
                if game_state.num_board_cards() == 0:
                    game_state.setup_postflop('flop')
                else:
                    game_state.setup_postflop('postflop')

        if hand:
            # Assume the hero got the the current game state with the given hand.
            game_state.update_current_hand(hand)

        # game_state.debug_output()

    def parallel_train(self, hands, hand_mapping, players, game_state, fast_forward_actions, mem_efficient=True, batch_size=psutil.cpu_count(logical=True)):
        manager = Manager()
        train_regret = manager.dict()
        train_strategy = manager.dict()
        hand_strategy_aggregate = manager.list()
        calculated = manager.dict()

        def process_batch(batch_hands):
            with Pool(processes = psutil.cpu_count(logical=True)) as pool:
                pool.starmap(
                    self.process_hand_wrapper,
                    [(hand, hand_mapping, players, game_state, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions) for hand in batch_hands]
                )

        num_batches = (len(hands) + batch_size - 1) // batch_size  # Calculate number of batches

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(hands))
            batch_hands = hands[start_index:end_index]

            process_batch(batch_hands)

            if mem_efficient:
                # Clear the regret and strategy sums to save memory. objects are reduplicated into each process
                self.regret_sum.clear()
                self.strategy_sum.clear()

                # Update with solved root nodes (does this do anything, idk yet! TODO figure out wth cfr is doing... will these values just be washed away without the following states being solved? probably... but this is better than nothing >:())
                self.regret_sum.update(dict(train_regret))
                self.strategy_sum.update(dict(train_strategy))

                gc.collect()

        return train_regret, train_strategy, hand_strategy_aggregate

    def process_hand_wrapper(self, hand, hand_mapping, players, game_state, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions):
        self.process_hand(hand, hand_mapping, players, game_state, train_regret, train_strategy, hand_strategy_aggregate, calculated, fast_forward_actions)

    cpdef train_realtime(self, GameState game_state):
        cdef double[:] probs
        cdef list hands = []
        cdef Deck deck = game_state.deck.clone()

        # clone the gamestate and silence the logging.
        cloned_game_state = game_state.clone()
        cloned_game_state.silent = True

        for iter_num in range(self.realtime_iterations):
            probs = np.ones(self.num_players, dtype=np.float64)
            self.cfr_traverse(cloned_game_state, probs, 0, self.cfr_realtime_depth, .005)

    # NOTE: I cant pickle lambdas so this is going here.
    cpdef default_double(self):
        return 0.0

    cdef double[:] cfr_traverse(self, GameState game_state, double[:] probs, int depth, int max_depth, float epsilon=0):
        
        cdef int cur_player_index = game_state.player_index
        cdef Player cur_player = game_state.players[cur_player_index]
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
        cdef list available_actions = game_state.players[game_state.player_index].get_available_actions(game_state)

        # If the game_state is terminal, the depth exceeds the max depth, or the current player has a negligible probability of reaching this state, return the utility.
        if game_state.is_terminal_river() or depth >= max_depth or probs[cur_player_index] < self.prune_probability_threshold:
            # TODO: Incorporate a mechanism to return bet chips to players if the current (potentially non-termonal) state is forced to enter a terminal state.
            game_state.progress_to_showdown()
            return self.calculate_utilities(game_state, game_state.winner_index)

        player_hash = cur_player.hash(game_state)

        self.regret_sum[player_hash] = (self.regret_sum.get(player_hash, defaultdict(self.default_double)), prune)

        util = {action: np.zeros(num_players, dtype=np.float64) for action in available_actions}
        strategy = self.get_strategy(available_actions, probs, game_state, cur_player, prune)

        if monte_carlo and available_actions:
            # Can we also prune the MC samples from the regret and strategy sums?
            epsilon_calc = epsilon
            rand_value = np.random.rand()
            if rand_value < epsilon_calc:
                uniform_prob = 1.0 / len(available_actions)
                for action in available_actions:
                    strategy[action] = uniform_prob
            
            strategy_list = np.array([strategy[a] for a in available_actions], dtype=np.float64)
            strategy_list /= strategy_list.sum()
            action_index = np.random.choice(len(available_actions), p=strategy_list)

            available_actions = available_actions[:action_index+1][-1:]


        for action in available_actions:
            new_game_state = game_state.clone()
            if new_game_state.handle_action(action):
                if new_game_state.num_board_cards() == 0:
                    new_game_state.setup_postflop('flop')
                else:
                    new_game_state.setup_postflop('postflop')

                # If we've reached a public terminal state, reset current probs to 1.
                probs[:] = 1.0
            
            new_probs[:] = probs
            new_probs[cur_player_index] *= strategy[action]

            util[action] = self.cfr_traverse(new_game_state, new_probs, depth + 1, max_depth, epsilon)
        
        ### NOTE: I want to just take the maximum expected value. Thanks yuliia!
        for i in range(num_players):
            node_util[i] = sum(strategy[action] * util[action][i] for action in available_actions)

            ### NOTE: Causes divergence. Tends to either over or under correct.
            ### TODO: Figure out how to take a max-possible-regret approach.
            # best_action = max(available_actions, key=lambda action: strategy[action] * util[action][i])
            # node_util[i] = util[best_action][i]
        
        for action in available_actions:
            regret = util[action][cur_player_index] - node_util[cur_player_index]

            # Here we want to incorporate the opponent's likelihood of reaching this state into our regret.
            opp_contribution = 1.0
            for i in range(num_players):
                if i != cur_player_index:
                    opp_contribution *= probs[i]
            
            self.regret_sum[player_hash][action] += opp_contribution * regret

        return node_util

    cdef double[:] calculate_utilities(self, GameState game_state, int winner_index):
        cdef int num_players = len(game_state.players)
        cdef double[:] utilities = np.zeros(num_players, dtype=np.float64)

        rake = game_state.pot * .03 # 3% rake
        rake = rake if rake < 4 else 4
        pot = game_state.pot - rake
        for i, p in enumerate(game_state.players):
            if i == winner_index:
                utilities[i] = (pot - p.tot_contributed_to_pot)
            else:
                utilities[i] = -(p.tot_contributed_to_pot)

        return utilities

    cdef dict get_strategy(self, list available_actions, double[:] probs, GameState game_state, Player player, bint prune):
        current_player = game_state.player_index
        player_hash = player.hash(game_state)
        
        self.strategy_sum[player_hash] = (self.strategy_sum.get(player_hash, defaultdict(float)), prune)

        if not available_actions:
            return {}
        
        strategy = {}
        normalization_sum = 0.0
        regrets = [max(self.regret_sum[player_hash][action], 0) for action in available_actions]
        
        for action, regret in zip(available_actions, regrets):
            strategy[action] = regret
            normalization_sum += regret
        
        if normalization_sum > 0:
            for action in strategy:
                strategy[action] /= normalization_sum
                self.strategy_sum[player_hash][action] += probs[current_player] * strategy[action]
        else:
            uniform_prob = 1 / len(available_actions)
            for action in strategy:
                strategy[action] = uniform_prob
                self.strategy_sum[player_hash][action] += probs[current_player] * uniform_prob
        
        return strategy

    cdef dict get_average_strategy(self, AIPlayer player, GameState game_state):
        average_strategy = {}
        game_state_hash = player.hash(game_state)
        normalization_sum = 0
        cur_gamestate_strategy = self.strategy_sum.get(game_state_hash, defaultdict(float))

        if not cur_gamestate_strategy:
            actions = player.get_available_actions(game_state)
            cur_gamestate_strategy = {action: 1/len(actions) for action in actions}

        adjusted_regrets = {action: max(value, 0) for action, value in cur_gamestate_strategy.items()}
        normalization_sum = sum(adjusted_regrets.values())

        if normalization_sum > 0:
            for action, value in adjusted_regrets.items():
                average_strategy[action] = value / normalization_sum
        else:
            num_actions = len(adjusted_regrets)
            for action in adjusted_regrets:
                average_strategy[action] = 1 / num_actions

        return average_strategy
