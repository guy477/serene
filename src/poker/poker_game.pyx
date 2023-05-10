#!python
#cython: language_level=3


import pandas as pd


################################################################################################################
############################ POKER GAME ########################################################################
################################################################################################################

cdef class PokerGame:

    def __init__(self, int num_players, int initial_chips, int num_ai_players, int small_blind, int big_blind, int cfr_iterations):
        
        self.players = [Player(initial_chips) for _ in range(num_players - num_ai_players)] + [AIPlayer(initial_chips, cfr_iterations, num_players, small_blind, big_blind) for _ in range(num_ai_players)]
        self.game_state = GameState(self.players, small_blind, big_blind)
        self.profit_loss = []
        self.position_pl = []

    cpdef play_game(self, int num_hands=1):

        self.profit_loss.append([i.chips for i in self.game_state.players])

        for _ in range(num_hands):
            
            

            self.game_state.setup_preflop()
            while(not self.game_state.handle_action()):
                continue
    
            self.game_state.setup_postflop("flop")
            while(not self.game_state.handle_action()):
                continue

            self.game_state.setup_postflop("turn")
            while(not self.game_state.handle_action()):
                continue

            self.game_state.setup_postflop("river")
            while(not self.game_state.handle_action()):
                continue

            # Determine the winner and distribute the pot
            self.game_state.showdown()
            self.profit_loss.append([i.chips for i in self.game_state.players])
            self.game_state.reset()

            # Update the dealer position
            self.game_state.dealer_position = (self.game_state.dealer_position + 1) % len(self.players)
        
        pd.DataFrame(self.profit_loss).to_csv("profit_loss overtime by player.csv")

################################################################################








