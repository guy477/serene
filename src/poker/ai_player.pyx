#!python
#cython: language_level=3

from collections import defaultdict
import random

cdef class AIPlayer(Player):
    # AI player-specific methods will be added here
    
    def __init__(self, int initial_chips, list bet_sizing, CFRTrainer cfr_trainer):
        super().__init__(initial_chips, bet_sizing)
        
        self.strategy_trainer = cfr_trainer

        self.initialize_regret_strategy()


    cpdef get_action(self, GameState game_state, int player_index):
        cdef object user_input
        cdef bint valid = 0
        cdef bint raize = 0

        
        
        while valid == 0:
            if not game_state.players[player_index].folded:
                
                display_game_state(game_state, player_index)
                
                # Since we're performing realtime searching, we want to reset the regret/strat mappings for the current state.
                self.strategy_trainer.train_realtime(game_state)

                average_strategy = self.strategy_trainer.get_average_strategy(self, game_state)
                
                #####
                print(average_strategy)
                #####
                
                actions, probabilities = zip(*average_strategy.items())

                # Choose an action based on the probability distribution
                user_input = random.choices(actions, probabilities, k=1)[0]

                #####
                print(user_input)
                #####


                # for the current implementation, we just want to min-raise. 
                if user_input[0] == "raise" or user_input[0] == "all-in":
                    self.take_action(game_state, player_index, user_input)
                    raize = 1
                    valid = 1
                elif user_input[0] == "fold" or user_input[0] == "call" or user_input[0] == "all-in":
                    self.take_action(game_state, player_index, user_input)
                    valid = 1
                else:
                    print("Invalid input. Please enter call, raise, or fold.")
            else:
                valid = 1
        return raize

    cdef initialize_regret_strategy(self):
        self.regret = <dict>defaultdict(lambda: 0)
        self.strategy_sum = <dict>defaultdict(lambda: 0)

    cpdef get_strategy(self, list available_actions, float[:] probs, GameState game_state):
        current_player = game_state.player_index
        game_state_hash = self.hash(game_state)

        strategy = {action: max(self.regret.get((game_state_hash, action), 0), 0) for action in available_actions}
        normalization_sum = sum(strategy.values())

        if normalization_sum > 0:
            for action in strategy:
                strategy[action] /= normalization_sum
                if self.strategy_sum.get((game_state_hash, action), 0) == 0:
                    self.strategy_sum[(game_state_hash, action)] = 0
                self.strategy_sum[(game_state_hash, action)] += probs[current_player] * strategy[action]
        else:
            num_actions = len(available_actions)
            for action in strategy:
                strategy[action] = 1 / num_actions
                if self.strategy_sum.get((game_state_hash, action), 0) == 0:
                    self.strategy_sum[(game_state_hash, action)] = 0
                self.strategy_sum[(game_state_hash, action)] += probs[current_player] * strategy[action]

        return strategy

    cpdef clone(self):
        # there has to be a better way to clone. Currently, im wasting memory by recreating a CFRTrainer object each time i cosntruct a new AIPlayer.
        # Better: I should reconsider how i handle the CFR object. Probably easiest to create this as part of the Poker_Game object and pass clones of it to each AIPlayer before each iteration? Update it after each iteration?
        cdef AIPlayer new_player = AIPlayer(self.chips, self.bet_sizing, self.strategy_trainer)
        new_player.hand = self.hand
        new_player.abstracted_hand = self.abstracted_hand

        new_player.regret = self.regret
        new_player.strategy_sum = self.strategy_sum

        new_player.folded = self.folded
        new_player.contributed_to_pot = self.contributed_to_pot
        new_player.tot_contributed_to_pot = self.tot_contributed_to_pot
        new_player.prior_gains = self.prior_gains

        return new_player