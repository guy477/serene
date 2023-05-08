#!python
#cython: language_level=3

import random
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

    cpdef train(self):
        # Main loop for the training iterations
        for _ in range(self.iterations):
            players = [Player(self.initial_chips) for _ in range(self.num_players)]
            game_state = GameState(players, self.initial_chips, self.small_blind, self.big_blind)
            game_state.fisher_yates_shuffle()
            game_state.deal_private_cards()
            game_state.deal_public_cards()
            print(self.strategy_sum)
            self.traverse_game_tree(game_state, 0, [1.0] * self.num_players)

    cpdef str get_best_action(self, GameState game_state, int player_index):
        # This method should implement the logic to get the best action based on the game state and CFR algorithm
        # For now, it returns a random action as a placeholder
        
        return random.choice(["call", "raise", "fold"])

    cpdef double traverse_game_tree(self, GameState game_state, int player_index, list reach_probabilities):
        if game_state.is_terminal():
            return game_state.get_payoff(player_index)

        current_player = game_state.get_current_player()
        information_set = game_state.get_information_set(current_player)
        available_actions = game_state.get_available_actions()

        cdef list strategy = self.get_strategy(information_set, current_player, available_actions)
        cdef list next_reach_probabilities = reach_probabilities.copy()
        cdef list action_utility = [0] * len(available_actions)

        for i, action in enumerate(available_actions):
            next_game_state = game_state.apply_action(action, current_player)
            next_reach_probabilities[current_player] *= strategy[i]
            action_utility[i] = -self.traverse_game_tree(next_game_state, player_index, next_reach_probabilities)

        node_utility = sum([strategy[i] * action_utility[i] for i in range(len(available_actions))])

        if current_player == player_index:
            for i, action in enumerate(available_actions):
                regret = action_utility[i] - node_utility
                self.regret_sum[information_set][i] += regret * reach_probabilities[current_player]
                self.strategy_sum[information_set][i] += strategy[i] * reach_probabilities[current_player]

        return node_utility

    cpdef list get_strategy(self, str information_set, int player_index, list available_actions):
        cdef list strategy = [max(self.regret_sum[information_set][i], 0) for i in range(len(available_actions))]
        cdef double normalizing_sum = sum(strategy)

        if normalizing_sum > 0:
            strategy = [strategy[i] / normalizing_sum for i in range(len(available_actions))]
        else:
            strategy = [1.0 / len(available_actions)] * len(available_actions)

        return strategy

