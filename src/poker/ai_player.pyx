#!python
#cython: language_level=3



cdef class AIPlayer(Player):
    # AI player-specific methods will be added here
    
    def __init__(self, int initial_chips, int cfr_iterations, int num_players, int small_blind, int big_blind):
        super().__init__(initial_chips)
        self.strategy_trainer = CFRTrainer(cfr_iterations, num_players, initial_chips, small_blind, big_blind)

    cpdef get_action(self, GameState game_state, int player_index):
        cdef str user_input
        cdef int bet_amount
        cdef bint valid = 0
        cdef bint raize = 0
        
        while valid == 0:
            if not game_state.players[player_index].folded:
                display_game_state(game_state, player_index)

                user_input = self.strategy_trainer.get_best_action(game_state, player_index)
                
                if user_input == "call":
                    self.player_action(game_state, player_index, "call")
                    valid = 1

                # for the current implementation, we just want to min-raise. 
                elif user_input == "raise":
                    bet_amount = int(game_state.current_bet)
                    self.player_action(game_state, player_index, "raise", bet_amount)
                    raize = 1
                    valid = 1
                elif user_input == "fold":
                    self.player_action(game_state, player_index, "fold")
                    valid = 1
                else:
                    print("Invalid input. Please enter call, raise, or fold.")
            else:
                valid = 1
        return raize