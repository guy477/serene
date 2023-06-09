#!python
#cython: language_level=3


import pandas as pd


################################################################################################################
############################ POKER GAME ########################################################################
################################################################################################################

cdef class PokerGame:

    def __init__(self, int num_players, int initial_chips, int num_ai_players, int small_blind, int big_blind, list bet_sizing, CFRTrainer cfr_trainer):
        
        self.players = [Player(initial_chips, bet_sizing) for _ in range(num_players - num_ai_players)] + [AIPlayer(initial_chips, bet_sizing, cfr_trainer) for _ in range(num_ai_players)]
        self.game_state = GameState(self.players, small_blind, big_blind, cfr_trainer.num_simulations)
        self.profit_loss = []
        
        self.position_pl = {'D':0, 'SB':0, 'BB':0, 'UTG':0, 'MP':0, 'CO':0}

    cpdef play_game(self, int num_hands=1):

        self.profit_loss.append([i.chips for i in self.game_state.players])

        for _ in range(num_hands):
            
            
            # the game is reset as a part of the preflop_setup.
            print('dealing preflop')
            print(self.game_state.debug_output())
            self.game_state.setup_preflop()
            print(self.game_state.debug_output())
            while(not self.game_state.handle_action()):
                print(self.game_state.debug_output())
                continue
        
            print('dealing flop')
            self.game_state.setup_postflop("flop")
            print(self.game_state.debug_output())
            while(not self.game_state.handle_action()):
                print(self.game_state.debug_output())
                continue

            print('dealing turn')
            self.game_state.setup_postflop("turn")
            print(self.game_state.debug_output())
            while(not self.game_state.handle_action()):
                print(self.game_state.debug_output())
                continue

            print('dealing river')
            self.game_state.setup_postflop("river")
            print(self.game_state.debug_output())
            while(not self.game_state.handle_action()):
                print(self.game_state.debug_output())
                continue

            print("Showdown")
            # Determine the winner and distribute the pot
            self.game_state.showdown()
            print(self.game_state.debug_output())

            # Log some statistics for debugging/analysis
            self.profit_loss.append([i.chips for i in self.game_state.players])
            for i in self.game_state.players:
                self.position_pl[i.position] += i.prior_gains - i.tot_contributed_to_pot
                #i.chips = 1000
            print(f"Player {self.game_state.winner_index + 1} wins the hand.")
            input("Press enter to continue to next hand.")

        pd.DataFrame(self.profit_loss).to_csv("results/profit_loss.csv")
        
        print(pd.DataFrame.from_dict(self.position_pl, orient='index', columns=['values'])/num_hands)

################################################################################








