#!python
#cython: language_level=3

import logging
import pandas as pd


################################################################################################################
############################ POKER GAME ########################################################################
################################################################################################################
cdef list SUITS = ['C', 'D', 'H', 'S']
cdef list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


cdef class PokerGame:
    def __init__(self, int num_players, int initial_chips, int num_ai_players, int small_blind, int big_blind, list bet_sizing, CFRTrainer cfr_trainer, list suits=SUITS, list values=VALUES):
        self.players = [Player(initial_chips, bet_sizing, True) for _ in range(num_players - num_ai_players)] + [Player(initial_chips, bet_sizing, False) for _ in range(num_ai_players)]
        self.game_state = GameState(self.players, small_blind, big_blind, cfr_trainer.num_simulations, False, suits, values)
        self.profit_loss = []
        self.position_pl = {position: 0 for position in self.game_state.positions}
        self.logger = logging.getLogger('PokerGame')
        self.logger.setLevel(logging.DEBUG)

    cpdef play_game(self, int num_hands=1):
        self.profit_loss.append([player.chips for player in self.game_state.players])

        for _ in range(num_hands):
            self.logger.debug('Dealing preflop')
            self.game_state.setup_preflop()
            self._play_round()
            
            self.logger.debug('Dealing flop')
            self.game_state.setup_postflop("flop")
            self._play_round()
            
            self.logger.debug('Dealing turn')
            self.game_state.setup_postflop("turn")
            self._play_round()
            
            self.logger.debug('Dealing river')
            self.game_state.setup_postflop("river")
            self._play_round()
            
            self.logger.debug("Showdown")
            self.game_state.showdown()
            
            self.profit_loss.append([player.chips for player in self.game_state.players])
            for player in self.game_state.players:
                self.position_pl[player.position] += player.prior_gains - player.tot_contributed_to_pot
            
            print(f"Player {self.game_state.winner_index} wins the hand.")
            input("Press enter to continue to next hand.")
        
        # pd.DataFrame(self.profit_loss).to_csv("profit_loss.csv")
        print(pd.DataFrame.from_dict(self.position_pl, orient='index', columns=['values']) / num_hands)

    cpdef _play_round(self):
        while not self.game_state.handle_action():
            self.logger.debug(self.game_state.debug_output())


