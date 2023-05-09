#!python
#cython: language_level=3

from collections import defaultdict

cdef class Player:
    def __init__(self, int initial_chips):
        self.chips = initial_chips
        self.hand = 0
        self.folded = False
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
        self.initialize_regret_strategy() 

    cpdef get_action(self, GameState game_state, int player_index):
        cdef str user_input
        cdef bint valid = 0
        cdef bint raize = 0
        
        while valid == 0:
            if not game_state.players[player_index].folded:
                display_game_state(game_state, player_index)

                user_input = self.get_user_input("Enter action (call, raise, fold): ")
                
                if user_input == "call":
                    self.take_action(game_state, player_index, "call")
                    valid = 1
                elif user_input == "raise":
                    self.take_action(game_state, player_index, "raise")
                    raize = 1
                    valid = 1
                elif user_input == "fold":
                    self.take_action(game_state, player_index, "fold")
                    valid = 1
                else:
                    print("Invalid input. Please enter call, raise, or fold.")
            else:
                valid = 1
        return raize

    cpdef take_action(self, GameState game_state, int player_index, str action, int bet_size = 0):
        bet_amount = max(game_state.big_blind, game_state.current_bet)
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
            player.tot_contributed_to_pot += call_amount
            player.folded = False

        elif action == "raise":
            if bet_size:
                bet_amount = bet_size
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

            if player.chips <= 0:
                print(f"Player {player_index + 1} is out of chips.")

        elif action == "fold":
            player.folded = True

        else:
            raise ValueError("Invalid action")

    cdef initialize_regret_strategy(self):
        self.regret = <dict>defaultdict(lambda: 0)
        self.strategy_sum = <dict>defaultdict(lambda: 0)

    cpdef get_strategy(self, list available_actions, float[:] probs, int current_player):
        strategy = {action: max(self.regret.get((current_player, action), 0), 0) for action in available_actions}
        normalization_sum = sum(strategy.values())

        if normalization_sum > 0:
            for action in strategy:
                strategy[action] /= normalization_sum
                self.strategy_sum[(current_player, action)] += probs[current_player] * strategy[action]
        else:
            num_actions = len(available_actions)
            for action in strategy:
                strategy[action] = 1 / num_actions
                if self.strategy_sum.get((current_player, action), 0) == 0:
                    self.strategy_sum[(current_player, action)] = 0
                self.strategy_sum[(current_player, action)] += probs[current_player] * strategy[action]

        return strategy

    cpdef add_card(self, unsigned long long card):
        self.hand |= card

    cpdef str get_user_input(self, prompt):
        return input(prompt)

    cpdef get_available_actions(self, GameState game_state, int player_index):
        return ['call', 'raise', 'fold']

    cpdef clone(self):
        cdef Player new_player = Player(self.chips)
        new_player.hand = self.hand
        new_player.folded = self.folded
        new_player.contributed_to_pot = self.contributed_to_pot
        new_player.tot_contributed_to_pot = self.contributed_to_pot
        return new_player

    cpdef reset(self):
        self.hand = 0
        self.folded = False
        self.contributed_to_pot = 0
        self.tot_contributed_to_pot = 0
