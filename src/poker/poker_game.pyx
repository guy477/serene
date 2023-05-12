#!python
#cython: language_level=3


import pandas as pd


################################################################################################################
############################ POKER GAME ########################################################################
################################################################################################################

cdef class PokerGame:

    def __init__(self, int num_players, int initial_chips, int num_ai_players, int small_blind, int big_blind, list bet_sizing, int cfr_iterations, int cfr_depth, int cfr_realtime_depth):
        
        self.players = [Player(initial_chips, bet_sizing) for _ in range(num_players - num_ai_players)] + [AIPlayer(initial_chips, bet_sizing, cfr_iterations, cfr_depth, cfr_realtime_depth, num_players, small_blind, big_blind) for _ in range(num_ai_players)]
        self.game_state = GameState(self.players, small_blind, big_blind)
        self.profit_loss = []
        
        self.position_pl = {'D':0, 'SB':0, 'BB':0, 'UTG':0, 'MP':0, 'CO':0}

    cpdef play_game(self, int num_hands=1):

        self.profit_loss.append([i.chips for i in self.game_state.players])

        for _ in range(num_hands):
            
            
            # the game is reset as a part of the preflop_setup.
            print("SETTING UP PREFLOP")
            self.game_state.setup_preflop()
            while(not self.game_state.handle_action()):
                continue

            print("SETTING UP FLOP")
            self.game_state.setup_postflop("flop")
            while(not self.game_state.handle_action()):
                continue

            print("SETTING UP TURN")
            self.game_state.setup_postflop("turn")
            while(not self.game_state.handle_action()):
                continue

            print("SETTING UP RIVER")
            self.game_state.setup_postflop("river")
            while(not self.game_state.handle_action()):
                continue

            print("Showdown")
            # Determine the winner and distribute the pot
            self.game_state.showdown()

            # Log some statistics for debugging/analysis
            self.profit_loss.append([i.chips for i in self.game_state.players])
            for i in self.game_state.players:
                self.position_pl[i.position] += i.prior_gains - i.tot_contributed_to_pot
                #i.chips = 1000
            print(f"Player {self.game_state.winner_index + 1} wins the hand.")
            

        pd.DataFrame(self.profit_loss).to_csv("results/profit_loss.csv")
        
        print(pd.DataFrame.from_dict(self.position_pl, orient='index', columns=['values'])/num_hands)

################################################################################








