# cython: language_level=3

import numpy as np
cimport numpy as np


from collections import defaultdict
from cython.view cimport array

cdef class CFR:
    
    def __init__(self, int cfr_depth, int prune_depth = 9999, double prune_probability = 1e-8):
        self.cfr_depth = cfr_depth

        self.prune_depth = prune_depth
        self.prune_probability_threshold = prune_probability


    cdef progress_gamestate_to_showdown(self, GameState game_state, LocalManager local_manager):
        ###
        ### wrapper; i considered performing monte-carlo sampling for a number of iterations
        ###    But i figured that would be too much too soon.
        ### TODO Investigate the feasibility of montecarlo terminal sampling
        ###

        cdef list available_actions
        
        # Get to first terminal state by sampling the average path.
        # while not (game_state.is_terminal() or game_state.is_terminal_river()):
        #     available_actions = game_state.get_current_player().get_available_actions(game_state)
        #     strategy = self.get_average_strategy(game_state.get_current_player(), game_state, local_manager)
        #     game_state.step(select_random_action(strategy))

        game_state.progress_to_showdown() # will call to terminal node (flop, turn, river, showdown)


    # TODO: Clean up this logic
    cdef double[:] cfr_traverse(self, GameState game_state, double[:] probs, int depth, LocalManager local_manager):
        cdef dict strategy
        cdef int action_index
        cdef double opp_contribution, regret
        cdef dict[double[:]] util

        cdef object action = ()

        cdef int num_players = len(game_state.players)
        cdef int cur_player_index = game_state.player_index
        cdef Player cur_player = game_state.get_current_player()
        cdef list player_hash = cur_player.hash(game_state)
        cdef list available_actions = game_state.get_current_player().get_available_actions(game_state)
        
        cdef double[:] node_util = array(shape=(num_players,), itemsize=sizeof(double), format="d")
        cdef double[:] new_probs = array(shape=(num_players,), itemsize=sizeof(double), format="d")

        cdef bint merge_criteria = depth == 0
        cdef bint prune_criteria = depth >= self.prune_depth
        cdef bint depth_criteria = not cur_player.folded

        regret_sum = local_manager.get_regret_sum().get_set(player_hash, defaultdict(self.default_double), prune_criteria, merge_criteria)
        strategy_sum = local_manager.get_strategy_sum().get_set(player_hash, defaultdict(self.default_double), prune_criteria, merge_criteria)

        if game_state.is_terminal_river() or depth >= self.cfr_depth or probs[cur_player_index] < self.prune_probability_threshold:
            self.progress_gamestate_to_showdown(game_state, local_manager)
            return self.calculate_utilities(game_state, game_state.winner_index)

        util = {action: np.zeros(num_players, dtype=np.float64) for action in available_actions}
        strategy = self.get_strategy(available_actions, probs, game_state, cur_player, local_manager)

        for action in available_actions:
            new_game_state = game_state.clone()
            
            if new_game_state.step(action):
                new_probs[:] = probs  
                # total = np.sum(probs)
                # if total > 0:
                #     new_probs[:] = probs               
                #     new_probs = new_probs / total 
                # else:
                #     for i in range(num_players):
                #         new_probs[i] = 1
            
            new_probs[cur_player_index] *= strategy[action]
            util[action] = self.cfr_traverse(new_game_state, new_probs, depth + depth_criteria, local_manager)

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

        rake = game_state.pot * .05 # 5% rake
        rake = min(rake//1, 3) # .3BB rake cap (500nl heads up)
        pot = game_state.pot - rake

        for i, p in enumerate(game_state.players):
            if i == winner_index:
                utilities[i] = (pot - p.tot_contributed_to_pot)
            else:
                utilities[i] = -(p.tot_contributed_to_pot)

        return utilities


    cdef dict get_strategy(self, list available_actions, double[:] probs, GameState game_state, Player player, LocalManager local_manager):
        cdef double normalization_sum, uniform_prob, regret, min_regret, total_positive_regrets
        cdef object player_hash, action
        cdef list regrets, regrets_norm
        cdef int current_player
        cdef dict strategy

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
        else: # all(x <= 0 for x in regrets):  # only negative regret; or pos+neg = 0
            min_regret = min(regrets)
            positive_regrets = [regret - min_regret for regret in regrets]
            total_positive_regrets = sum(positive_regrets)
            for action, positive_regret in zip(available_actions, positive_regrets):
                strategy[action] = positive_regret / total_positive_regrets
                strategy_sum[action] += probs[current_player] * strategy[action]
        
        return strategy


    cpdef dict get_average_strategy(self, Player player, GameState game_state, LocalManager local_manager):
        cdef object average_strategy, game_state_hash, cur_gamestate_strategy, action
        cdef double uniform_prob, regret, normalization_sum
        cdef list available_actions, regrets, regrets_norm
        cdef int current_player

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
        else: # all([x <= 0 for x in regrets]):  # all(x <= 0 for x in regrets):  # only negative regret; or pos+neg = 0
            # Convert negative regrets to positive values by adding the minimum regret value to each
            min_regret = min(regrets)
            positive_regrets = [regret - min_regret for regret in regrets]
            total_positive_regrets = sum(positive_regrets)
            for action, positive_regret in zip(average_strategy.keys(), positive_regrets):
                average_strategy[action] = positive_regret / total_positive_regrets

        return average_strategy

    cpdef double default_double(self):
        ### Idk why this is necessary. 
        ### creating "float"s in cython seem to have half precision.
        ###     could be wrong. 
        return 0.0
