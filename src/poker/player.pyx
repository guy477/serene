#!python
#cython: language_level=3


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
        self.expected_hand_strength = 1
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
        


    cpdef public void assign_position(self, str position, int player_index):
        self.player_index = player_index
        self.position = position


    cpdef public bint get_action(self, GameState game_state, int player_index):
        cdef str user_input
        cdef float raise_amount
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
                    user_input = self.get_user_input("Enter sizing: " + str(self.bet_sizing[game_state.cur_round_index]) + " : ")
                    try:
                        raise_amount = float(user_input)
                    except Exception as e:
                        print(str(e))
                        continue
                    self.take_action(game_state, player_index, ("raise", raise_amount))
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

    cpdef public bint take_action(self, GameState game_state, int player_index, object action):
        cdef Player player = game_state.players[player_index]
        cdef int call_amount
        cdef bint raize = False

        # We want the current betting history to have the player's position as well.
        # This can probably be restricted to the first round if complexity in future states becomes too much of an issue.
        game_state.betting_history[game_state.cur_round_index].append((self.position, action))

        if action[0] == "call":
            call_amount = game_state.current_bet - player.contributed_to_pot
            if call_amount > player.chips:
                call_amount = player.chips

            player.chips = player.chips - call_amount
            game_state.pot += call_amount
            player.contributed_to_pot += call_amount
            player.tot_contributed_to_pot += call_amount

        elif action[0] == "raise" or action[0] == "all-in":
            if action[0] == "raise":
                bet_amount = int((action[1]) * (game_state.pot))
                raize = True
            else:
                bet_amount = player.chips

            if bet_amount < game_state.current_bet:
                if bet_amount != player.chips:
                    raise ValueError("Raise amount must be at least equal to the current bet or an all-in.")

            if bet_amount > player.chips:
                bet_amount = player.chips

            player.chips -= (bet_amount)
            game_state.pot += (bet_amount)
            game_state.current_bet = (bet_amount + player.contributed_to_pot)
            player.contributed_to_pot += (bet_amount)
            player.tot_contributed_to_pot += (bet_amount)
            

        elif action[0] == "blinds":
            bet_amount = action[1]

            if bet_amount > player.chips:
                bet_amount = player.chips

            player.chips -= (bet_amount)
            game_state.pot += (bet_amount)
            player.contributed_to_pot += (bet_amount)
            
            ### NOTE: by ignoring the forced contributions we don't negatively impact the player's regret calculation.
            ### TODO: is there a better place to do this for consistency? or is this the exact reason i made blinds distinct from raises? reality may have never known.
            # player.tot_contributed_to_pot += (bet_amount)

            game_state.current_bet = bet_amount

        elif action[0] == "fold":
            player.folded = True

        else:
            raise ValueError("Invalid action")

        return raize


    cpdef public void add_card(self, unsigned long long card):
        self.hand |= card

    cpdef public str get_user_input(self, prompt):
        return input(prompt)

    cpdef list get_available_actions(self, GameState game_state):
        if game_state.is_terminal_river():
            return [('call', 0)]

        # Initialize the list of possible actions with call and fold
        cdef list ret = [('call', 0), ('fold', 0)]
        cdef int current_bet = game_state.current_bet
        cdef int cur_round_index = game_state.cur_round_index
        cdef int pot = game_state.pot
        cdef int chips = self.chips
        cdef int bet_to_call = current_bet - self.contributed_to_pot
        cdef float i

        # If the player has folded only force repeated folds vs no-ops 
        if self.folded:
            return [('call', 0)]

        # If player chips is 0 we can assume all in (TODO REVISIT WHEN SPLIT POTS)
        if chips <= 0:
            return [('check', 0)]


        # If there is no action, disallow folding.
        if bet_to_call == 0:
            ret.remove(('fold', 0))

        # Allow All-ins if the gamestate's current bet size is "significant" relative to our stack.
        if current_bet >= (chips // 3):
            ret.append(('all-in', 0))

        # Prevent open limping. Inefficient.
        if game_state.cur_round_index == 0 and (len(game_state.betting_history[0]) == 2 or all([('fold', 0) == x[1] for x in game_state.betting_history[0][2:]])):
            ret.remove(('call', 0))

        # Check if the player can cover the current bet.
        if chips >= current_bet:
            
            for i in self.bet_sizing[cur_round_index]:
                raise_amount = int(pot * i)
                if chips >= (current_bet + raise_amount) and raise_amount > current_bet and chips > raise_amount:
                    # Represent the raise as a proportion rather than the actual amount.
                    ret.append(('raise', i))

            # Better guide preflop raises.
            if game_state.cur_round_index == 0 and ('raise', 1.5) in game_state.betting_history[0]:
                ret.remove(('raise', 1.5))
            elif game_state.cur_round_index == 0 and ('raise', 1.5) not in game_state.betting_history[0] and ('raise', 5) in ret:
                ret.remove(('raise', 5))
        else:
            ret.remove(('call', 0))
        
        assert(len(ret) > 0)

        return ret

    cpdef public Player clone(self):
        cdef Player new_player = Player(self.chips, self.bet_sizing)
        new_player.hand = self.hand
        
        new_player.position = self.position
        new_player.player_index = self.player_index

        new_player.expected_hand_strength = self.expected_hand_strength
        new_player.abstracted_hand = self.abstracted_hand
        new_player.folded = self.folded
        new_player.contributed_to_pot = self.contributed_to_pot
        new_player.tot_contributed_to_pot = self.tot_contributed_to_pot
        new_player.prior_gains = self.prior_gains
        return new_player

    cpdef public void reset(self):
        self.hand = 0
        self.abstracted_hand = ''
        self.folded = False
        # if self.chips == 0:
        self.expected_hand_strength = 1
        self.chips = 1000
        self.position = ''
        self.player_index = 0
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
        self.prior_gains = 0
    
    cpdef public str hash(self, GameState game_state):
        # hsh = hash((self.abstracted_hand, game_state.board, self.position, game_state.cur_round_index, str(self.betting_history)))
        #hsh = (self.abstracted_hand, self.position, game_state.cur_round_index, str(game_state.betting_history[0]))
        # hsh = (self.abstracted_hand, game_state.board, self.position, game_state.cur_round_index, str(game_state.betting_history))
        # hsh = (self.abstracted_hand, self.position, game_state.cur_round_index, tuple(self.get_available_actions(game_state)), str(game_state.betting_history[0]))
        
        ### NOTE we want to get abstracted hands here
        ### NOTE NOTE this hsh is further hashed in the HashTable object.
        hsh = str((self.abstracted_hand, self.position, game_state.pot//200, tuple(self.get_available_actions(game_state)))) #  str(game_state.betting_history)
        return hsh
