#!python
#cython: language_level=3

import random
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import itertools
from collections import defaultdict

cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef class CFRTrainer:
    def __init__(self, int iterations, int realtime_iterations, int num_simulations, int cfr_depth, int cfr_realtime_depth, int num_players, int initial_chips, int small_blind, int big_blind, list bet_sizing, list suits=SUITS, list values=VALUES):
        self.iterations = iterations
        self.realtime_iterations = realtime_iterations
        self.num_simulations = num_simulations
        self.cfr_realtime_depth = cfr_realtime_depth
        self.cfr_depth = cfr_depth

        self.num_players = num_players
        self.initial_chips = initial_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.bet_sizing = bet_sizing

        self.suits = suits
        self.values = values

        self.regret_sum = {}
        self.strategy_sum = {}

    cpdef train(self):
        cdef float[:] probs

        # Generate all possible 2-card hands
        all_possible_hands = [(r1 + r2 + 's') for r1 in self.values for r2 in self.values if r1 != r2 and (self.values.index(r1) > self.values.index(r2))]  # Suited hands
        all_possible_hands += [(r1 + r2 + 'o') for r1 in self.values for r2 in self.values if (self.values.index(r1) >= self.values.index(r2))]  # Offsuit hands
        all_possible_hands = sorted(all_possible_hands)

        # Generate all possible cards
        cards = [r + s for r in reversed(self.values) for s in self.suits]

        # Generate all possible hands (for testing, we can sample this to focus on the first n to maximize iteration potential)
        hands = list(sorted(itertools.combinations(cards, 2)))

        # Define a new game state for training
        players = [AIPlayer(self.initial_chips, self.bet_sizing, self) for _ in range(self.num_players)]
        
        # Ensure the gamestate is silenced
        game_state = GameState(players, self.small_blind, self.big_blind, self.num_simulations, True, self.suits, self.values)

        for iter_num in range(self.iterations):
            epsilon = .1  # Exploration parameter

            for hand in hands:
                print(f'Iteration: {iter_num}; Hand: {hand}', end = '\r')
                game_state.setup_preflop(hand)

                probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
                probs[:] = 1  # Initialize probabilities

                self.cfr_traverse(game_state, probs, 0, self.cfr_depth, epsilon)
                ###input("Press Enter to continue to the next hand...")

        game_state.dealer_position = 4
        game_state.setup_preflop()

        preflop_range = []
        for player_idx in range(len(players)):
            player = players[player_idx]
            if player_idx > 0:
                last_player = players[player_idx - 1]
                game_state.load_custom_betting_history(0, (last_player.position, ('fold', 0)))

            for hand in all_possible_hands:
                player.abstracted_hand = hand
                strategy = self.get_average_strategy(player, game_state)
                preflop_range.append((player.position, hand, strategy))

        return preflop_range

    cpdef train_realtime(self, GameState game_state):
        cdef float[:] probs
        cdef list hands = []
        cdef list deck = game_state.deck[:]

        # clone the gamestate and silence the logging.
        cloned_game_state = game_state.clone()
        cloned_game_state.silent = True

        for i in range(len(cloned_game_state.players)):
            if i != cloned_game_state.player_index:
                hands.append(cloned_game_state.players[i].hand)
                cloned_game_state.deck.extend(hand_to_cards(hands[-1]))
                cloned_game_state.players[i].hand = 0



        for iter_num in range(self.realtime_iterations):
            ###print(f'Realtime Iteration: {iter_num}')

            probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
            probs[:] = 1  # Initialize probabilities

            self.cfr_traverse(cloned_game_state, probs, 0, self.cfr_realtime_depth, .1)
            ###input("Press Enter to continue to the next iteration...")



    cdef cfr_traverse(self, GameState game_state, float[:] probs, int depth, int max_depth, float epsilon=0):
        cdef int num_players = len(game_state.players)
        cdef float[:] new_probs = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")
        cdef float[:] node_util = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")

        if game_state.is_terminal_river() or depth >= max_depth:
            game_state.showdown()
            return self.calculate_utilities(game_state, game_state.winner_index)

        current_player = game_state.player_index
        player_hash = game_state.players[current_player].hash(game_state)

        ###print(f"Current player hash: {player_hash}")

        if self.regret_sum.get(player_hash, {}) == {}:
            self.regret_sum[player_hash] = {}

        available_actions = game_state.players[current_player].get_available_actions(game_state, current_player)
        strategy = self.get_strategy(available_actions, probs, game_state, game_state.players[current_player])

        util = defaultdict(lambda: cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f"))
        monte_carlo = depth > 4

        if monte_carlo:
            if available_actions:
                if np.random.rand() < epsilon:
                    strategy = {action: 1/len(available_actions) for action in available_actions}
                    strategy_list = [1/len(available_actions) for _ in available_actions]
                else:
                    strategy_list = [strategy[a] for a in available_actions]
                action_index = np.random.choice(range(len(available_actions)), p=strategy_list)
                action = available_actions[action_index]
                available_actions = [action]

        for action in available_actions:
            ###print(f"Action: {action}")
            #print(f"__________\nCURRENT GAMESTATE")
            #game_state.debug_output()
            new_game_state = game_state.clone()
            #print(f"__________\nCLONED GAMESTATE")
            #new_game_state.debug_output()
            if new_game_state.handle_action(action):
                if new_game_state.num_board_cards() == 0:
                    new_game_state.setup_postflop('flop')
                else:
                    new_game_state.setup_postflop('postflop')
            
            ###print(f"__________\nITER GAMESTATE")
            ###new_game_state.debug_output()

            ###input('verify game state...')

            for i in range(num_players):
                new_probs[i] = probs[i] * strategy[action] if i == current_player else probs[i]

            ###print(f"Before traverse, action: {action}, probs: {list(new_probs)}")
            util[action] = self.cfr_traverse(new_game_state, new_probs, depth + 1, max_depth, epsilon)
            ###print(f"After traverse, action: {action}, util: {list(util[action])}")
            ###input(f"Press Enter to continue with action {action}...")

        for i in range(num_players):
            node_util[i] = sum(strategy[action] * util[action][i] for action in available_actions)

        for action in available_actions:
            regret = util[action][current_player] - node_util[current_player]
            ###print(f"Regret for action {action}: {regret}")
            self.regret_sum[player_hash][action] = self.regret_sum[player_hash].get(action, 0) + regret

        ###print(f"Node utility: {list(node_util)}")
        ###print(f"Updated regret sums: {self.regret_sum[player_hash]}")
        ###input("Press Enter to continue to the next node...")

        return node_util

    cdef float[:] calculate_utilities(self, GameState game_state, int winner_index):
        cdef int num_players = len(game_state.players)
        cdef float[:] utilities = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")

        pot = game_state.pot

        for i, p in enumerate(game_state.players):
            if i == winner_index:
                # The winner gets the pot added to their initial chips
                utilities[i] = pot - p.tot_contributed_to_pot
            else:
                # Other players have their total contribution deducted
                utilities[i] = -p.tot_contributed_to_pot

        return utilities
        ###print(f"Utilities: {list(utilities)}")
        ###input("Press Enter to continue to the next calculation...")


    cpdef get_strategy(self, list available_actions, float[:] probs, GameState game_state, Player player):
        current_player = game_state.player_index
        player_hash = player.hash(game_state)
        
        if self.strategy_sum.get(player_hash, {}) == {}:
            self.strategy_sum[player_hash] = {}
        
        if not available_actions:
            return {}
        
        # Calculate the strategy based on regret sums
        strategy = {}
        normalization_sum = 0.0
        
        for action in available_actions:
            regret = max(self.regret_sum.get(player_hash, {}).get(action, 0), 0)
            strategy[action] = regret
            normalization_sum += regret
        
        # Normalize the strategy
        if normalization_sum > 0:
            for action in strategy:
                strategy[action] /= normalization_sum
                self.strategy_sum[player_hash][action] = self.strategy_sum[player_hash].get(action, 0) + probs[current_player] * strategy[action]
        else:
            num_actions = len(available_actions)
            for action in available_actions:
                strategy[action] = 1 / num_actions
                self.strategy_sum[player_hash][action] = self.strategy_sum[player_hash].get(action, 0) + probs[current_player] * strategy[action]
        
        return strategy

        ###print(f"Strategy: {strategy}")
        ###input("Press Enter to continue to the next strategy calculation...")

    cdef get_average_strategy(self, AIPlayer player, GameState game_state):
        average_strategy = {}
        game_state_hash = player.hash(game_state)
        normalization_sum = 0
        cur_gamestate_strategy = self.strategy_sum.get(game_state_hash, {})

        if not cur_gamestate_strategy:
            actions = player.get_available_actions(game_state, player.player_index)
            cur_gamestate_strategy = {action: 1/len(actions) for action in actions}

        for action, value in cur_gamestate_strategy.items():
            normalization_sum += value

        if normalization_sum > 0:
            for action, value in cur_gamestate_strategy.items():
                average_strategy[action] = value / normalization_sum
        else:
            num_actions = len(cur_gamestate_strategy)
            for action in cur_gamestate_strategy:
                average_strategy[action] = 1 / num_actions

        ###print(f"Average strategy: {average_strategy}")
        ###input("Press Enter to continue to the next average strategy calculation...")

        return average_strategy
