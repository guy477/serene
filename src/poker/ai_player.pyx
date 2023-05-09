#!python
#cython: language_level=3

from collections import defaultdict
import random

cdef class AIPlayer(Player):
    # AI player-specific methods will be added here
    
    def __init__(self, int initial_chips, int cfr_iterations, int num_players, int small_blind, int big_blind):
        super().__init__(initial_chips)
        self.strategy_trainer = CFRTrainer(cfr_iterations, num_players, initial_chips, small_blind, big_blind)
        self.initialize_regret_strategy() 
    cpdef get_action(self, GameState game_state, int player_index):
        cdef str user_input
        cdef bint valid = 0
        cdef bint raize = 0

        
        
        while valid == 0:
            if not game_state.players[player_index].folded:
                
                display_game_state(game_state, player_index)

                self.strategy_trainer.train_on_game_state(game_state, iterations=self.strategy_trainer.iterations)

                average_strategy = self.strategy_trainer.get_average_strategy(self, game_state)
                
                #####
                print(average_strategy)
                #####
                
                actions, probabilities = zip(*average_strategy.items())

                # Choose an action based on the probability distribution
                user_input = random.choices(actions, probabilities, k=1)[0]
                
                if user_input == "call":
                    self.take_action(game_state, player_index, "call")
                    valid = 1

                # for the current implementation, we just want to min-raise. 
                elif user_input == "raise":
                    self.take_action(game_state, player_index, "raise")
                    raize = 1
                    valid = 1
                elif user_input == "fold":
                    self.take_action(game_state, player_index, "fold")
                    valid = 1
                else:
                    print("Invalid input. Please enter call, raise, or fold.")
            else:
                valid = 1
        return raize

    cdef initialize_regret_strategy(self):
        self.regret = <dict>defaultdict(lambda: 0)
        self.strategy_sum = <dict>defaultdict(lambda: 0)

    cpdef get_strategy(self, list available_actions, float[:] probs, int current_player):
        strategy = {action: max(self.regret.get((current_player, action), 0), 0) for action in available_actions}
        normalization_sum = sum(strategy.values())

        if normalization_sum > 0:
            for action in strategy:
                strategy[action] /= normalization_sum
                self.strategy_sum[(current_player, action)] += probs[current_player] * strategy[action]
        else:
            num_actions = len(available_actions)
            for action in strategy:
                strategy[action] = 1 / num_actions
                if self.strategy_sum.get((current_player, action), 0) == 0:
                    self.strategy_sum[(current_player, action)] = 0
                self.strategy_sum[(current_player, action)] += probs[current_player] * strategy[action]

        return strategy

    cpdef clone(self):
        cdef AIPlayer new_player = AIPlayer(self.chips, self.strategy_trainer.iterations, self.strategy_trainer.num_players, self.strategy_trainer.small_blind, self.strategy_trainer.big_blind)
        new_player.hand = self.hand
        new_player.folded = self.folded
        new_player.contributed_to_pot = self.contributed_to_pot
        new_player.tot_contributed_to_pot = self.tot_contributed_to_pot
        new_player.strategy_trainer = self.strategy_trainer
        return new_player