#!python
#cython: language_level=3

import random
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from collections import defaultdict

cdef class CFRTrainer:

    def __init__(self, int iterations, int num_players, int initial_chips, int small_blind, int big_blind):
        self.iterations = iterations
        self.num_players = num_players
        self.initial_chips = initial_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.regret_sum = <dict>defaultdict(lambda: [0] * num_players)
        self.strategy_sum = <dict>defaultdict(lambda: [0] * num_players)

    cpdef str get_best_action(self, GameState game_state, int player_index):
        # This method should implement the logic to get the best action based on the game state and CFR algorithm
        # For now, it returns a random action as a placeholder
        
        return random.choice(["call", "raise", "fold"])



    cpdef train(self):
        # Main loop for the training iterations
        cdef float[:] probs

        for _ in range(self.iterations):
            print(f'iterations: {_}', end = '\r')
            players = [AIPlayer(self.initial_chips, self.iterations, self.num_players, self.small_blind, self.big_blind) for _ in range(self.num_players)]
            game_state = GameState(players, self.small_blind, self.big_blind)
            game_state.setup_preflop()
            probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
            for i in range(len(probs)):
                probs[i] = 1
            self.cfr_traverse(game_state, self.num_players, probs, 0, 2)
            game_state.reset()
            

    cpdef train_on_game_state(self, GameState game_state, int iterations):
        cdef float[:] probs
        for _ in range(iterations):
            print(f'iterations: {_}', end = '\r')
            probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
            for i in range(len(probs)):
                probs[i] = 1
            self.cfr_traverse(game_state, game_state.player_index, probs, 0, 5)

    cdef cfr_traverse(self, GameState game_state, int player, float[:] probs, int depth, int max_depth):
        cdef int num_players = len(game_state.players)
        cdef float[:] new_probs = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")
        cdef float[:] node_util = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")

        # Base case: Check if the game has reached a terminal state
        if game_state.is_terminal_river() or depth >= max_depth:
            # Calculate and return the utility for each player
            
            game_state.showdown()
            return self.calculate_utilities(game_state, player)

        # Otherwise, continue traversing the game tree
        current_player = game_state.player_index

        # Get available actions for the current player
        # Implement a function to get available actions
        available_actions = game_state.players[current_player].get_available_actions(game_state, current_player)
        strategy = game_state.players[current_player].get_strategy(available_actions, probs, current_player)
        
        util = defaultdict(lambda: cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f", mode="w"))

        # Iterate through available actions
        for action in available_actions:
            # Create a copy of the game state and apply the action
            new_game_state = game_state.clone()
            
            # if after the action taken we move to the next game state, go ahead and progress the round.
            if (new_game_state.handle_action(action)):
                if new_game_state.num_board_cards() == 0:
                    new_game_state.setup_postflop('flop')
                else:
                    new_game_state.setup_postflop('postflop')

            # Update the probability distribution for the new game state
            for i in range(num_players):
                if i == current_player:
                    new_probs[i] = probs[i] * strategy[action]
                else:
                    new_probs[i] = probs[i]

            util[action] = self.cfr_traverse(new_game_state, player, new_probs, depth + 1, max_depth)

            for i in range(num_players):
                node_util[i] += strategy[action] * util[action][i]
        
        if current_player == player:
            for action in available_actions:
                regret = util[action][player] - node_util[player]
                game_state.players[current_player].regret[(current_player, action)] += regret

        return node_util

    cdef float[:] calculate_utilities(self, GameState game_state, int player):
        cdef int num_players = len(game_state.players)
        cdef float[:] utilities = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")

        # Get the current pot size
        # the pot is reset in showdown - need to reconsider this logic. 
        pot = game_state.pot

        # Check if there is only one active player (i.e., all other players have folded)
        remaining_players = sum([not p.folded for p in game_state.players])
        
        if remaining_players == 1:
            for i, p in enumerate(game_state.players):
                if not p.folded:
                    # The remaining player wins the pot
                    utilities[i] = pot - p.tot_contributed_to_pot
                else:
                    # Players who folded lose their contributed chips
                    utilities[i] = -p.tot_contributed_to_pot
        else:
    
            for i, p in enumerate(game_state.players):
                if i == game_state.winner_index:
                    # The winner gains the pot minus their contributed chips
                    utilities[i] = pot - p.tot_contributed_to_pot
                else:
                    # Non-winning players lose their contributed chips
                    utilities[i] = -p.tot_contributed_to_pot
                    
        return utilities
    
    cdef get_average_strategy(self, AIPlayer player, GameState game_state):
        average_strategy = {}
        normalization_sum = 0

        for (player_idx, action), value in player.strategy_sum.items():
            if player_idx == game_state.player_index:
                normalization_sum += value

        if normalization_sum > 0:
            for (player_idx, action), value in player.strategy_sum.items():
                if player_idx == game_state.player_index:
                    average_strategy[action] = value / normalization_sum
        else:
            num_actions = len(player.strategy_sum)
            for (player_idx, action), value in player.strategy_sum.items():
                if player_idx == game_state.player_index:
                    average_strategy[action] = 1 / num_actions

        return average_strategy