#!python
#cython: language_level=3







################################################################################################################
############################ POKER GAME ########################################################################
################################################################################################################

cdef class PokerGame:

    def __init__(self, int num_players, int initial_chips, int num_ai_players, int small_blind, int big_blind, int cfr_iterations):
        self.players = [Player(initial_chips) for _ in range(num_players - num_ai_players)] + [AIPlayer(initial_chips, cfr_iterations, num_players, small_blind, big_blind) for _ in range(num_ai_players)]
        self.game_state = GameState(self.players, initial_chips, small_blind, big_blind)

    cpdef play_game(self, int num_hands=1):

        for _ in range(num_hands):
            
            # Betting rounds
            self.game_state.handle_blinds()
            self.game_state.deal_private_cards()
            self.game_state.preflop()
            self.game_state.postflop("flop")
            self.game_state.postflop( "turn")
            self.game_state.postflop( "river")

            # Determine the winner and distribute the pot
            self.game_state.showdown()
            self.game_state.reset()

            # Update the dealer position
            self.game_state.dealer_position = (self.game_state.dealer_position + 1) % len(self.players)

################################################################################








