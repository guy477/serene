#!python
#cython: language_level=3

import logging
import pandas as pd
import time

################################################################################################################
############################ POKER GAME ########################################################################
################################################################################################################
cdef list SUITS = ['C', 'D', 'H', 'S']
cdef list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


cdef class PokerGame:
    def __init__(self, int num_players, int initial_chips, int num_ai_players, int small_blind, int big_blind, list bet_sizing, CFRTrainer cfr_trainer, LocalManager local_manager, list suits=SUITS, list values=VALUES):
        self.players = [Player(initial_chips, bet_sizing, True) for _ in range(num_players - num_ai_players)] + [Player(initial_chips, bet_sizing, False) for _ in range(num_ai_players)]
        self.game_state = GameState(self.players, small_blind, big_blind, cfr_trainer.num_simulations, False, suits, values)
        
        self.strategy_trainer = cfr_trainer
        self.local_manager = local_manager
        
        self.profit_loss = []
        self.position_pl = {position: 0 for position in self.game_state.positions}
        self.logger = logging.getLogger('PokerGame')
        self.logger.setLevel(logging.DEBUG)

    cpdef play_game(self, int num_hands=1):
        self.profit_loss.append([player.chips for player in self.game_state.players])

        for _ in range(num_hands):
            print('*** Dealing Preflop ***')
            self.game_state.setup_preflop()
            display_game_state(self.game_state, self.game_state.player_index)
            
            self._play_round()

            if self.game_state.is_terminal_river():
                self.skip_to_showdown()
                continue

            print('*** Dealing Flop ***')
            self._play_round()
            
            if self.game_state.is_terminal_river():
                self.skip_to_showdown()
                continue


            print('*** Dealing turn ***')
            self._play_round()
            
            if self.game_state.is_terminal_river():
                self.skip_to_showdown()
                continue

            
            print('Dealing River')
            self._play_round()
            
            self.skip_to_showdown()
        
        # pd.DataFrame(self.profit_loss).to_csv("profit_loss.csv")
        print(pd.DataFrame.from_dict(self.position_pl, orient='index', columns=['values']) / num_hands)

    cpdef _play_round(self):
        while not self.game_state.step(self.get_action()):
            display_game_state(self.game_state, self.game_state.player_index)

        display_game_state(self.game_state, self.game_state.player_index)


     
    cpdef get_action(self):
            if self.game_state.get_current_player().is_human:
                action = self.game_state.get_current_player().get_action(self.game_state)
                    
            elif not self.game_state.get_current_player().folded:
                fast_forward_actions = build_fast_forward_actions(self.game_state.action_space)
                player_hand_str_tuple = card_tuple_to_str_tuple(ulong_to_card_tuple(self.game_state.get_current_player().hand))

                local_manager = self.strategy_trainer.train(self.local_manager, [fast_forward_actions], [player_hand_str_tuple])
                
                strategy = self.strategy_trainer.get_average_strategy(self.game_state.get_current_player(), self.game_state, local_manager)

                print(strategy)
                action = select_random_action(strategy)
            else:
                action = ('call', 0)
            return action


    cpdef skip_to_showdown(self):
        print("Showdown")
        self.game_state.progress_to_showdown()
        
        self.profit_loss.append([player.chips for player in self.game_state.players])
        for player in self.game_state.players:
            self.position_pl[player.position] += player.prior_gains - player.tot_contributed_to_pot

        display_game_state(self.game_state, self.game_state.player_index)
        
        print(f"\nPlayer {self.game_state.players[self.game_state.winner_index].position} wins the hand.")
        for i in range(5):
            print(f"Starting next hand in {5-i} seconds...", end = '\r')
            time.sleep(1)