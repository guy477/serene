#!python
#cython: language_level=3

from collections import defaultdict


cdef class Player:
    def __init__(self, int initial_chips, list bet_sizing):
        self.chips = initial_chips
        self.bet_sizing = bet_sizing

        self.hand = 0
        self.abstracted_hand = ''

        self.folded = False
        self.position = ''
        self.player_index = 0
        self.prior_gains = 0
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
        self.betting_history = [[],[],[],[]]


    cpdef assign_position(self, str position, int player_index):
        self.player_index = player_index
        self.position = position


    cpdef get_action(self, GameState game_state, int player_index):
        cdef str user_input
        cdef bint valid = 0
        cdef bint raize = 0
        
        while valid == 0:
            if not game_state.players[player_index].folded:
                display_game_state(game_state, player_index)

                user_input = self.get_user_input("Enter action (call, raise, all-in, fold): ")
                
                if user_input == "call":
                    self.take_action(game_state, player_index, ("call", 0))
                    valid = 1
                elif user_input == "raise":
                    user_input = self.get_user_input("Enter sizing: " + str(self.bet_sizing[game_state.cur_round_index]))
                    try:
                        user_input = float(user_input)
                    except Exception as e:
                        continue
                    self.take_action(game_state, player_index, ("raise", user_input))
                    raize = 1
                    valid = 1
                elif user_input == "all-in":
                    self.take_action(game_state, player_index, ("all-in", 0))
                    raize = 1
                    valid = 1
                elif user_input == "fold":
                    self.take_action(game_state, player_index, ("fold", 0))
                    valid = 1
                else:
                    print("Invalid input. Please enter call, raise, or fold.")
            else:
                valid = 1
        return raize

    cpdef take_action(self, GameState game_state, int player_index, object action):
        bet_amount = max(game_state.big_blind, game_state.current_bet)
        cdef Player player = game_state.players[player_index]
        cdef int call_amount

        if player.folded or player.chips <= 0:
            return


        # Keep tabs on the betting history for each round.        
        self.betting_history[game_state.cur_round_index].append(action)
        
        if action[0] == "call":
            call_amount = game_state.current_bet - player.contributed_to_pot
            if call_amount > player.chips:
                call_amount = player.chips

            player.chips -= call_amount
            game_state.pot += call_amount
            player.contributed_to_pot += call_amount
            player.tot_contributed_to_pot += call_amount
            player.folded = False

        elif action[0] == "raise":
            bet_amount = (action[1]) * game_state.pot
            if bet_amount < game_state.current_bet:
                if bet_amount != player.chips:
                    raise ValueError("Raise amount must be at least equal to the current bet or an all-in.")
            
            call_amount = game_state.current_bet - player.contributed_to_pot
            if call_amount + bet_amount > player.chips:
                bet_amount = player.chips - call_amount

            player.chips -= (call_amount + bet_amount)
            game_state.pot += (call_amount + bet_amount)
            player.contributed_to_pot += (call_amount + bet_amount)
            player.tot_contributed_to_pot += (call_amount + bet_amount)
            game_state.current_bet += bet_amount
            player.folded = False

            # if player.chips <= 0:
            #     print(f"Player {player_index + 1} is out of chips.")
        elif action[0] == "blinds":
            bet_amount = action[1]
            if bet_amount < game_state.current_bet:
                if bet_amount != player.chips:
                    raise ValueError("Raise amount must be at least equal to the current bet or an all-in.")
            
            call_amount = game_state.current_bet - player.contributed_to_pot
            if call_amount + bet_amount > player.chips:
                bet_amount = player.chips - call_amount

            player.chips -= (call_amount + bet_amount)
            game_state.pot += (call_amount + bet_amount)
            player.contributed_to_pot += (call_amount + bet_amount)
            player.tot_contributed_to_pot += (call_amount + bet_amount)
            game_state.current_bet += bet_amount
            player.folded = False

        elif action[0] == "all-in":
            bet_amount = player.chips
            if bet_amount < game_state.current_bet:
                if bet_amount != player.chips:
                    raise ValueError("Raise amount must be at least equal to the current bet or an all-in.")
            
            call_amount = game_state.current_bet - player.contributed_to_pot
            if call_amount + bet_amount > player.chips:
                bet_amount = player.chips - call_amount

            player.chips -= (call_amount + bet_amount)
            game_state.pot += (call_amount + bet_amount)
            player.contributed_to_pot += (call_amount + bet_amount)
            player.tot_contributed_to_pot += (call_amount + bet_amount)
            game_state.current_bet += bet_amount
            player.folded = False

        elif action[0] == "fold":
            player.folded = True

        else:
            raise ValueError("Invalid action")


    cpdef add_card(self, unsigned long long card):
        self.hand |= card

    cpdef str get_user_input(self, prompt):
        return input(prompt)

    cpdef get_available_actions(self, GameState game_state, int player_index):
        ret = [('call', 0), ('fold', 0)]
        cdef Player player = game_state.players[player_index]
        

        if player.chips < 0 or player.folded:
            return []

        if player.chips > game_state.current_bet:
            for i in self.bet_sizing[game_state.cur_round_index]:
                if player.chips > game_state.pot * i and game_state.pot * i > game_state.current_bet:
                    # we dont want to represent the raise as the actual amount, that way the CFR mapping knows what it's looking at.
                    ret.append(('raise', i))

            # if the current player's chips is greater than the current bet, they can go all-in.
            ret.append(('all-in', 0))
        

        if game_state.current_bet == 0 or player.contributed_to_pot == game_state.current_bet:
            ret.remove(('fold', 0))

        return ret

    cpdef clone(self):
        cdef Player new_player = Player(self.chips, self.bet_sizing)
        new_player.hand = self.hand
        
        new_player.position = self.position
        new_player.player_index = self.player_index

        new_player.betting_history = self.betting_history[:]

        new_player.abstracted_hand = self.abstracted_hand
        new_player.folded = self.folded
        new_player.contributed_to_pot = self.contributed_to_pot
        new_player.tot_contributed_to_pot = self.contributed_to_pot
        new_player.prior_gains = self.prior_gains
        return new_player

    cpdef reset(self):
        self.hand = 0
        self.abstracted_hand = ''
        self.folded = False
        # if self.chips == 0:
        self.chips = 1000
        self.position = ''
        self.player_index = 0
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
        self.prior_gains = 0
        self.betting_history = [[],[],[],[]]
    
    cpdef hash(self, GameState game_state):
        # hsh = hash((self.abstracted_hand, game_state.board, self.position, game_state.cur_round_index, str(self.betting_history)))
        hsh = (self.abstracted_hand, game_state.board, self.position, game_state.cur_round_index, str(self.betting_history))
        return hsh
