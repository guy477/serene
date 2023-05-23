#!python
#cython: language_level=3

import random
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import itertools
from collections import defaultdict




cdef class CFRTrainer:

    def __init__(self, int iterations, int realtime_iterations, int num_simulations, int cfr_depth, int cfr_realtime_depth, int num_players, int initial_chips, int small_blind, int big_blind, list bet_sizing):
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

        self.regret_sum = {}
        self.strategy_sum = {}


    cpdef train(self):
        # Main loop for the training iterations
        cdef float[:] probs

        # just two suits
        suits = ['C', 'S']

        # Define all possible ranks
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

        # Generate all possible 2-card hands
        all_possible_hands = [(r1 + r2 + 's') for r1 in ranks for r2 in ranks if r1 != r2 and (ranks.index(r1) > ranks.index(r2))]  # Suited hands
        all_possible_hands += [(r1 + r2 + 'o')  for r1 in ranks for r2 in ranks if (ranks.index(r1) >= ranks.index(r2))]  # Offsuit hands
        all_possible_hands = sorted(all_possible_hands)

        # Generate all possible cards
        cards = [r + s for r in reversed(ranks) for s in suits]

        # Generate all possible hands (for testing, we can sample this to focus on the first n to maximize iteration potential)
        hands = list(sorted(itertools.combinations(cards, 2)))

        # Define the a new gamestate for training.
        players = [AIPlayer(self.initial_chips, self.bet_sizing, self) for _ in range(self.num_players)]
        game_state = GameState(players, self.small_blind, self.big_blind, self.num_simulations)

        for _ in range(self.iterations):
            # define exploitation parameter; 10% of the time we explore regardless of the strategy values.
            epsilon = .1

            # Guided Training; for each iteration we consider all abstracted preflop positions.
            for hand in hands:
                print(f'iterations: {_}; Hand {hand}', end = '\r')
                
                game_state.setup_preflop(hand)

                probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
                for i in range(len(probs)):
                    probs[i] = 1
                
                self.cfr_traverse(game_state, probs, 0, self.cfr_depth, epsilon)

        # reset the dealer position so that player index 0 is first to act.
        game_state.dealer_position = 4
        game_state.setup_preflop()


        # extract un-opened preflop ranges
        preflop_range = []
        for player_idx in range(len(players)):
            player = players[player_idx]
            if player_idx > 0:
                last_player = players[player_idx - 1]
                game_state.load_custom_betting_history(0, (last_player.position, ('fold', 0)))

            for hand in all_possible_hands:
                player.abstracted_hand = hand
                strategy = self.get_average_strategy(player, game_state)
                preflop_range += [(player.position, hand, strategy)]
        
        return preflop_range


    cpdef train_realtime(self, GameState game_state):
        cdef float[:] probs
        cdef list hands = []
        cdef list deck = game_state.deck[:]

        # Create a copy of the villians' hands and clear their values. During a realtime search we do not know this information.
        # We also want to re-add the villians' hands to the current game_state deck (while keeping a duplicate of it).
        for i in range(len(game_state.players)):
            if not i == game_state.player_index:
                hands.append(game_state.players[i].hand)
                game_state.deck.extend(hand_to_cards(hands[-1]))
                game_state.players[i].hand = 0
        
        # Traverse the gametree for however many iterations are specified
        for _ in range(self.realtime_iterations):
            print(f'iterations: {_}', end = '\r')

            probs = cython.view.array(shape=(self.num_players,), itemsize=sizeof(float), format="f")
            for i in range(len(probs)):
                probs[i] = 1
            
            self.cfr_traverse(game_state.clone(), probs, 0, self.cfr_realtime_depth, .1)

        # reassign private information.
        for i in range(len(game_state.players)):
            if not i == game_state.player_index:
                game_state.players[i].hand = hands.pop(0)

        # reset potential public information
        game_state.deck = deck[:]
        

    cdef cfr_traverse(self, GameState game_state, float[:] probs, int depth, int max_depth, float epsilon = 0):
        cdef int num_players = len(game_state.players)
        cdef float[:] new_probs = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")
        cdef float[:] node_util = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")
        
        # Base case: Check if the game has reached a terminal state
        if game_state.is_terminal_river() or depth >= max_depth:
            # Calculate and return the utility for each player
            game_state.showdown()
            return self.calculate_utilities(game_state, game_state.winner_index)

        # Determine the current traversing player and get their relative game_state hash.
        # NOTE: For the realtime search, the hash will not map to any trained position since the private information is unavailable.
        #           Still trying to figure a way to deal with this.
        current_player = game_state.player_index
        player_hash = game_state.players[current_player].hash(game_state)

        if self.regret_sum.get(player_hash, {}) == {}:
            self.regret_sum[player_hash] = {}

        # Get the traversing player's available actions.
        available_actions = game_state.players[current_player].get_available_actions(game_state, current_player)
        strategy = self.get_strategy(available_actions, probs, game_state, game_state.players[current_player])

        util = defaultdict(lambda: cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f"))

        # Brain blast idea: use vanilla CFR up to a depth of n; then switch to MCCFR to evaluate to the end of the decision space
        monte_carlo = depth > 4

        # Take sample of the action space.
        if monte_carlo:
            if not available_actions == []:
                if np.random.rand() < epsilon:
                    # Explore
                    strategy = {action: 1/len(available_actions) for action in available_actions}
                    strategy_list = [1/len(available_actions) for _ in available_actions]

                else:
                    # Choose based on current probability space
                    strategy_list = [strategy[a] for a in available_actions]

                # Choose action
                action_index = np.random.choice(range(len(available_actions)), p=strategy_list)

                # Retrieve the action from the list
                action = available_actions[action_index]
                
                available_actions = [action]

    
        # Iterate through available actions
        for action in available_actions:
            # Create a copy of the game state and apply the action
            new_game_state = game_state.clone()
            
            # if after the action taken we move to the next game state, go ahead and progress the round.
            # the handle_action function will return a call to is_terminal
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
            
            # Recursively call this function with current_player as the traversing player. 
            util[action] = self.cfr_traverse(new_game_state, new_probs, depth + 1, max_depth, epsilon)
            

        
        # Update node util based on recursive results.
        for i in range(num_players):
            node_util[i] = sum(strategy[action] * util[action][i] for action in available_actions)

        for action in available_actions:
            regret = util[action][current_player] - node_util[current_player]
            # sort of hacky; defaultdicts are not respected in cython.
            # if the key has not been seen before, the uninitialized mapping will have a value 0.
            if self.regret_sum[player_hash].get(action, 0) == 0:
                self.regret_sum[player_hash][action] = 0
            self.regret_sum[player_hash][action] += regret



        return node_util

    cdef float[:] calculate_utilities(self, GameState game_state, int winner_index):
        cdef int num_players = len(game_state.players)
        cdef float[:] utilities = cython.view.array(shape=(num_players,), itemsize=sizeof(float), format="f")

        # Get the current pot size
        pot = game_state.pot

        # Check if there is only one active player (i.e., all other players have folded)
        remaining_players = sum([not p.folded for p in game_state.players])
        
        # TODO: add simulation code from ccluster to estimate hand probs based on player's hand range based on 
        #       preflop ranges 
        if remaining_players == 1:
            for i, p in enumerate(game_state.players):
                if not p.folded:
                    # The remaining player wins the pot
                    utilities[i] = (pot * p.expected_hand_strength - p.tot_contributed_to_pot) 
                else:
                    # Players who folded lose their contributed chips
                    utilities[i] = -p.tot_contributed_to_pot * (p.expected_hand_strength)
        else:
    
            for i, p in enumerate(game_state.players):
                #if i == winner_index:
                    # The winner gains the pot minus their contributed chips
                    utilities[i] = (pot * p.expected_hand_strength - p.tot_contributed_to_pot)
                # else:
                #     # Non-winning players lose their contributed chips;;;  + (p.chips if p.folded else 0)
                #     utilities[i] = -p.tot_contributed_to_pot
        
        return utilities

    cpdef get_strategy(self, list available_actions, float[:] probs, GameState game_state, Player player):
        current_player = game_state.player_index
        player_hash = player.hash(game_state)
        if self.strategy_sum.get(player_hash, {}) == {}:
            self.strategy_sum[player_hash] = {}     

        if available_actions == []:
            return {}

        strategy = {action: max(self.regret_sum.get(player_hash, {}).get(action, 0), 0) for action in available_actions}
        normalization_sum = sum(strategy.values())
        
        
        if normalization_sum > 0:
            for action in strategy:
                # initialize nested mapping if necessary
                if self.strategy_sum[player_hash].get(action, 0) == 0:
                    self.strategy_sum[player_hash][action] = 0

                strategy[action] /= normalization_sum
                self.strategy_sum[player_hash][action] += probs[current_player] * strategy[action]
            
        # either the regrets are all zero, or there are no regrets. In the case the regrets are all zero, we want to distribute the probabilities accordingly.
        else:
            strategy = {action: self.regret_sum.get(player_hash, {}).get(action, 0) for action in available_actions}
            min_regret = min(strategy.values())
        
            for action in strategy:
                strategy[action] -= min_regret

            normalization_sum = sum(strategy.values())

            # regrets are negative
            if normalization_sum > 0:
                for action in strategy:
                    if self.strategy_sum[player_hash].get(action, 0) == 0:
                        self.strategy_sum[player_hash][action] = 0
                    strategy[action] /= normalization_sum
                    self.strategy_sum[player_hash][action] += probs[current_player] * strategy[action]
            # regrets are all zero
            else:
                num_actions = len(available_actions)
                for action in strategy:
                    # initialize nested mapping if necessary
                    if self.strategy_sum[player_hash].get(action, 0) == 0:
                        self.strategy_sum[player_hash][action] = 0
                    
                    strategy[action] = 1 / num_actions
                    self.strategy_sum[player_hash][action] += probs[current_player] * strategy[action]


        return strategy    

    cdef get_average_strategy(self, AIPlayer player, GameState game_state):
        average_strategy = {}
        game_state_hash = player.hash(game_state)
        normalization_sum = 0
        cur_gamestate_strategy = self.strategy_sum.get(game_state_hash, {})

        if cur_gamestate_strategy == {}:
            actions = player.get_available_actions(game_state, player.player_index)
            cur_gamestate_strategy = {action: 1/len(actions) for action in actions}
        cur_gamestate_strategy = cur_gamestate_strategy.items()

        for action, value in cur_gamestate_strategy:
            normalization_sum += value

        if normalization_sum > 0:
            for action, value in cur_gamestate_strategy:
                average_strategy[action] = value / normalization_sum
        else:
            num_actions = len(cur_gamestate_strategy)
            for action, value in cur_gamestate_strategy:
                average_strategy[action] = 1 / num_actions

        return average_strategy