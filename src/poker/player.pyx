#!python
#cython: language_level=3

cdef class Player:
    def __init__(self, int initial_chips):
        self.chips = initial_chips
        self.hand = 0
        self.folded = False
        self.contributed_to_pot = 0

    cpdef get_action(self, GameState game_state, int player_index):
        cdef str user_input
        cdef int bet_amount
        cdef bint valid = 0
        cdef bint raize = 0
        
        while valid == 0:
            if not game_state.players[player_index].folded:
                display_game_state(game_state, player_index)

                user_input = self.get_user_input("Enter action (call, raise, fold): ")
                
                if user_input == "call":
                    self.player_action(game_state, player_index, "call")
                    valid = 1
                elif user_input == "raise":
                    bet_amount = int(self.get_user_input("Enter raise amount: "))
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

    cpdef player_action(self, GameState game_state, int player_index, str action, int bet_amount=0):
        cdef Player player = game_state.players[player_index]
        cdef int call_amount

        if player.folded or player.chips <= 0:
            return

        if action == "call":
            call_amount = game_state.current_bet - player.contributed_to_pot
            if call_amount > player.chips:
                call_amount = player.chips

            player.chips -= call_amount
            game_state.pot += call_amount
            player.contributed_to_pot += call_amount
            player.folded = False

            if player.chips <= 0:
                print(f"Player {player_index + 1} is out of chips.")

        elif action == "raise":
            if bet_amount < game_state.current_bet:
                if bet_amount != player.chips:
                    raise ValueError("Raise amount must be at least equal to the current bet or an all-in.")
            
            call_amount = game_state.current_bet - player.contributed_to_pot
            if call_amount + bet_amount > player.chips:
                bet_amount = player.chips - call_amount

            player.chips -= (call_amount + bet_amount)
            game_state.pot += (call_amount + bet_amount)
            player.contributed_to_pot += (call_amount + bet_amount)
            game_state.current_bet += bet_amount
            player.folded = False

            if player.chips <= 0:
                print(f"Player {player_index + 1} is out of chips.")

        elif action == "fold":
            player.folded = True

        else:
            raise ValueError("Invalid action")

    cpdef add_card(self, unsigned long long card):
        self.hand |= card

    cpdef str get_user_input(self, prompt):
        return input(prompt)

    cpdef reset(self):
        self.hand = 0
        self.folded = False
        self.contributed_to_pot = 0
