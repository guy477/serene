#!python
#cython: language_level=3

import random
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import itertools
from collections import defaultdict
from tqdm import tqdm

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
        
        # Ensure the game state is set up for preflop
        game_state = GameState(players, self.small_blind, self.big_blind, self.num_simulations, True, self.suits, self.values)

        train_regret = {}
        train_strategy = {}

        hand_strategy_aggregate = []  # List to store aggregated strategies for each hand
        calculated = {}

        for hand in hands:
            if calculated.get(hand_mapping[hand], False):
                continue
            calculated[hand_mapping[hand]] = True
            print('__________')
            print(f'Current Hand: {hand}', end = '\r')
            for iter_num in tqdm(range(self.iterations)):
                epsilon = (0.9995 ** iter_num)  # Update epsilon based on iteration  # .9998 for 20000 iterations
                #print(f'Iteration: {iter_num} for hand: {hand}', end='\r')

                # Initialize probabilities
                probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
                probs[:] = 1
                # #print(f'Current Hand: {hand}')
                game_state.setup_preflop(hand)

                ### NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
                ### NOTE  CONTINUE INVESTIGATING PROBS NOTE
                ### NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE

                # Perform CFR traversal for the current hand
                self.cfr_traverse(game_state.clone(), probs, 0, self.cfr_depth, epsilon)
                # input()

            # After iterations for a hand, calculate average strategy
            player_idx = 1  # Assuming you iterate over all players here
            player = players[player_idx]
            strategy = self.get_average_strategy(player, game_state)
            # show player hash and other game state info
            player_hash = player.hash(game_state)
            print(player_hash)
            print(f"Regret sum: {self.regret_sum[player_hash]}")
            print(f"Average strategy for hand {hand_mapping[hand]}: {strategy}")

            train_regret[player_hash] = self.regret_sum[player_hash]
            train_strategy[player_hash] = self.strategy_sum[player_hash]
            
            # Reset memory for current node.
            self.regret_sum = {}
            self.strategy_sum = {}
            
            # print(strategy)
            # input()
        
        # Set the global regret_sum and strategy_sum to the training values
        self.regret_sum = train_regret
        self.strategy_sum = train_strategy

        # After iterations for a hand, calculate average strategy
        player_idx = 1  # Assuming you iterate over all players here
        player = players[player_idx]
        calculated = {}

        for hand in hands:
            if calculated.get(hand_mapping[hand], False):
                continue
            calculated[hand_mapping[hand]] = True
            # Set player's abstracted hand based on the mapping
            player.abstracted_hand = hand_mapping[hand]

            # Prepend an Open-raise from the small blind:
            # game_state.load_custom_betting_history(0, ('SB', ('raise', 2.5)))

            strategy = self.get_average_strategy(player, game_state)
            #print(player.hash(game_state))

            #print(f'Average strategy for hand {hand_mapping[hand]}: {strategy}')

            # Append strategy to aggregate list
            hand_strategy_aggregate.append((player.position, hand_mapping[hand], strategy))

        return hand_strategy_aggregate




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
            ####print(f'Realtime Iteration: {iter_num}')

            probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
            probs[:] = 1  # Initialize probabilities

            self.cfr_traverse(cloned_game_state, probs, 0, self.cfr_realtime_depth, .005)
            ####input("Press Enter to continue to the next iteration...")



    cdef cfr_traverse(self, GameState game_state, float[:] probs, int depth, int max_depth, float epsilon=0):
        cdef int num_players = len(game_state.players)
        cdef float[:] node_util = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")

        #print(f"Entering cfr_traverse: depth={depth}, max_depth={max_depth}")
        #input("Press enter to continue.")
        
        if game_state.is_terminal_river() or depth >= max_depth:
            #print("Terminal state or max depth reached.")
            game_state.showdown()
            return self.calculate_utilities(game_state, game_state.winner_index)

        current_player = game_state.player_index
        player_hash = game_state.players[current_player].hash(game_state)

        if player_hash not in self.regret_sum:
            self.regret_sum[player_hash] = {action: 0 for action in game_state.players[current_player].get_available_actions(game_state)}
            #print(f"Initialized regret_sum for player_hash={player_hash}")
            #input("Press enter to continue.")

        available_actions = game_state.players[current_player].get_available_actions(game_state)
        #print(f"Available actions: {available_actions}")
        #input("Press enter to continue.")
        
        util = {action: cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f") for action in available_actions}
        strategy = self.get_strategy(available_actions, probs, game_state, game_state.players[current_player])
        #print(f"Strategy: {strategy}")
        #input("Press enter to continue.")

        monte_carlo = depth > 8
        #print(f"Monte Carlo: {monte_carlo}")
        #input("Press enter to continue.")

        # Monte Carlo sampling with epsilon exploration
        if monte_carlo:
            epsilon_calc = epsilon * .9 ** (1 + depth)
            rand_value = np.random.rand()
            #print(f"Random value for epsilon exploration: {rand_value}")
            #input("Press enter to continue.")

            if rand_value < epsilon:
                strategy = {action: 1 / len(available_actions) for action in available_actions}
            
            strategy_list = [strategy[a] for a in available_actions]
            if len(strategy_list) == 0:
                action_index = 0
                available_actions = []
            else:
                action_index = np.random.choice(range(len(available_actions)), p=strategy_list)
                #print(f"Strategy list: {strategy_list}"
            
                action = available_actions[action_index]
                available_actions = [action]
            #print(f"Chosen action in Monte Carlo: {action}")
            #input("Press enter to continue.")

        for action in available_actions:
            #print(f"Processing action: {action}")
            #input("Press enter to continue.")
            
            new_game_state = game_state.clone()
            if new_game_state.handle_action(action):
                #print(f"Action {action} handled, setting up postflop.")
                if new_game_state.num_board_cards() == 0:
                    new_game_state.setup_postflop('flop')
                else:
                    new_game_state.setup_postflop('postflop')

            new_probs = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")
            new_probs[:] = probs[:]
            new_probs[current_player] *= strategy[action]
            
            #print(f"New probabilities: {list(new_probs)}")
            #input("Press enter to continue.")
            
            util[action] = self.cfr_traverse(new_game_state, new_probs, depth + 1, max_depth, epsilon)
            #print(f"Utility for action {action}: {list(util[action])}")
            #input("Press enter to continue.")

        for i in range(num_players):
            node_util[i] = sum(strategy[action] * util[action][i] for action in available_actions)
            #print(f"Node utility for player {i}: {node_util[i]}")
            #input("Press enter to continue.")

        for action in available_actions:
            regret = util[action][current_player] - node_util[current_player]
            opp_contribution = np.prod(probs) / (probs[current_player] if probs[current_player] != 0 else 1)
            self.regret_sum[player_hash][action] += opp_contribution * regret

            #print(f"Updated regret_sum for player_hash={player_hash}, action={action}")
            #print(f"Regret: {regret}")
            #print(f"Opp contribution: {opp_contribution}")
        #print(f"Regret sum: {self.regret_sum[player_hash]}")
            #input("Press enter to continue.")


        #print(f"Returning node_util: {list(node_util)}")
        #input("Press enter to continue.")
        
        return node_util







    cdef float[:] calculate_utilities(self, GameState game_state, int winner_index):
        cdef int num_players = len(game_state.players)
        cdef float[:] utilities = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")

        # https://www.acrpoker.eu/real-money/poker-rake-calculation/
        rake = game_state.pot * .98
        rake = rake if rake < 4 else 4
        pot = game_state.pot - rake
        for i, p in enumerate(game_state.players):
            if i == winner_index:
                # The winner gets the pot added to their initial chips
                utilities[i] = (pot - p.tot_contributed_to_pot)
            else:
                # Other players have their total contribution deducted
                utilities[i] = -(p.tot_contributed_to_pot)

        return utilities


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



    cdef get_average_strategy(self, AIPlayer player, GameState game_state):
        average_strategy = {}
        game_state_hash = player.hash(game_state)
        normalization_sum = 0
        cur_gamestate_strategy = self.strategy_sum.get(game_state_hash, {})

        if not cur_gamestate_strategy:
            actions = player.get_available_actions(game_state)
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

        ####print(f"Average strategy: {average_strategy}")
        ####input("Press Enter to continue to the next average strategy calculation...")

        return average_strategy
