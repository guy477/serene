#!python
#cython: language_level=3

cdef class Player:
    def __init__(self, int initial_chips, list bet_sizing, bint is_human):
        self.is_human = is_human

        self.chips = initial_chips
        self.bet_sizing = bet_sizing

        self.hand = 0

        self.folded = False
        self.position = ''
        self.player_index = 0
        self.prior_gains = 0
        self.expected_hand_strength = 1
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
        


    cpdef void assign_position(self, str position, int player_index):
        self.player_index = player_index
        self.position = position


    cpdef object get_action(self, GameState game_state):
        cdef str user_input
        cdef float raise_amount
        cdef object action
        cdef bint valid = 0

        while valid == 0:
            if not self.folded:
                user_input = self.get_user_input("Enter action (call, raise, all-in, fold): ")
                
                if user_input == "call":
                    action = ("call", 0)
                    valid = 1
                elif user_input == "raise":
                    user_input = self.get_user_input("Enter sizing: " + str(self.bet_sizing[game_state.cur_round_index]) + " : ")
                    try:
                        raise_amount = float(user_input)
                    except Exception as e:
                        print(str(e))
                        continue
                    action = ("raise", raise_amount)
                    valid = 1
                elif user_input == "all-in":
                    action = ("all-in", 0)
                    valid = 1
                elif user_input == "fold":
                    action = ("fold", 0)
                    valid = 1
                else:
                    print("Invalid input.")
            else:
                valid = 1
                action = ('call', 0)
        
        return action

    cpdef bint take_action(self, GameState game_state, object action):
        cdef int call_amount
        cdef bint raize = False

        # We want the current betting history to have the player's position as well.
        # This can probably be restricted to the first round if complexity in future states becomes too much of an issue.
        game_state.action_space[game_state.cur_round_index].append((self.position, action))

        if self.folded:
            return 0

        if action[0] == "call":
            call_amount = game_state.current_bet - self.contributed_to_pot
            if call_amount > self.chips:
                call_amount = self.chips

            self.chips = self.chips - call_amount
            game_state.pot += call_amount
            self.contributed_to_pot += call_amount
            self.tot_contributed_to_pot += call_amount

        elif action[0] == "raise" or action[0] == "all-in":

            if action[0] == "raise":
                bet_amount = int((action[1]) * (game_state.pot))
                
            else:
                bet_amount = self.chips

            if bet_amount < game_state.current_bet:
                if bet_amount != self.chips:
                    game_state.debug_output()
                    raise ValueError("Raise amount must be at least equal to the current bet or an all-in.")
            else:
                raize = True

            if bet_amount > self.chips:
                bet_amount = self.chips

            self.chips -= (bet_amount)
            game_state.pot += (bet_amount)
            game_state.current_bet = (bet_amount + self.contributed_to_pot)
            self.contributed_to_pot += (bet_amount)
            self.tot_contributed_to_pot += (bet_amount)
            

        elif action[0] == "blinds":
            bet_amount = action[1]

            if bet_amount > self.chips:
                bet_amount = self.chips

            self.chips -= (bet_amount)
            game_state.pot += (bet_amount)
            self.contributed_to_pot += (bet_amount)
            
            ### NOTE: by ignoring the forced contributions we don't negatively impact the self's regret calculation.
            ### TODO: is there a better place to do this for consistency? or is this the exact reason i made blinds distinct from raises? reality may have never known.
            self.tot_contributed_to_pot += (bet_amount)

            game_state.current_bet = bet_amount

        elif action[0] == "fold":
            self.folded = True

        else:
            raise ValueError(f"Invalid action {action}.")

        return raize


    cpdef void add_card(self, unsigned long long card):
        self.hand |= card

    cpdef str get_user_input(self, prompt):
        return input(prompt)

    cpdef list get_available_actions(self, GameState game_state):
        if game_state.is_terminal_river():
            return [('call', 0)]

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
            return [('call', 0)]


        # If there is no action, disallow folding.
        if bet_to_call == 0:
            ret.remove(('fold', 0))

        # Allow All-ins if the gamestate's current bet size is "significant" relative to our stack.
        if current_bet >= (chips // 3):
            ret.append(('all-in', 0))

        
        if chips >= current_bet:
            ## Add raises that dont put us all in
            for i in self.bet_sizing[cur_round_index]:
                raise_amount = int(pot * i)
                # If we have enough chips to raise, the suggested raise is strictly larger than the current gamestate bet + our current contribution to the pot.
                if chips > raise_amount and (raise_amount > (current_bet + self.contributed_to_pot)):
                    # Represent the raise as a proportion rather than the actual amount.
                    ret.append(('raise', i))
            
            if ret[-1][0] != 'raise' and current_bet < (chips//3):
                ## then we know the only bet we can make is all-in
                ret.append(('all-in', 0))
        else:
            ret.remove(('call', 0))

        
        ### Handle preflop cases
        if game_state.cur_round_index == 0:
            
            if ('call', 0) in ret and (game_state.active_players() != 2 or all(['fold' == x[1][0] for x in game_state.action_space[0][2:]])):
                ## If call is in the return list and we're last to act... (this logic can be cleaned up.)
                if game_state.last_raiser == -1:
                    ret.remove(('call', 0))    # Prevent Limping

                elif game_state.last_raiser != game_state.player_index:
                    ret.remove(('call', 0))    # prevent calling with people left to act

            ## If no one has opened, only allow 2.25bb open
            if ('raise', 2.0) in ret and ('raise', 1.5) not in [x[1] for x in game_state.action_space[0]]:
                ret.remove(('raise', 2.0))
                
            ## If someone's opened, remove open raise (1.5x) 
            elif ('raise', 1.5) in ret and ('raise', 1.5) in [x[1] for x in game_state.action_space[0]]:
                ret.remove(('raise', 1.5))

        
        assert(len(ret) > 0)

        return ret

    cpdef Player clone(self):
        cdef Player new_player = Player(self.chips, self.bet_sizing, self.is_human)
        new_player.hand = self.hand
        
        new_player.position = self.position
        new_player.player_index = self.player_index

        new_player.expected_hand_strength = self.expected_hand_strength
        new_player.folded = self.folded
        new_player.contributed_to_pot = self.contributed_to_pot
        new_player.tot_contributed_to_pot = self.tot_contributed_to_pot
        new_player.prior_gains = self.prior_gains
        return new_player

    cpdef void reset(self):
        self.hand = 0
        self.folded = False
        # if self.chips == 0:
        self.expected_hand_strength = 1
        self.chips = 1000

        ## Assigned by gamestate on initialization
        # self.position = ''
        # self.player_index = 0
        
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
        
        ### NOTE: This shouldnt accumulate to insane values, right?
        # self.prior_gains = 0
    
    cpdef object hash(self, GameState game_state):
        ### NOTE/TODO: Add "player type" that can be referenced in LocalManager to give a unique regret/strategy for the unique player type.
        hsh = (self.hand, game_state.board, self.position, game_state.cur_round_index, tuple(self.get_available_actions(game_state)), tuple(game_state.action_space))
        
        return hsh
