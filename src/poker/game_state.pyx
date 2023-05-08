#!python
#cython: language_level=3

from libc.stdlib cimport rand, srand

cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


cdef class GameState:
    def __init__(self, list players, int initial_chips, int small_blind, int big_blind):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_position = 0
        self.pot = 0
        self.current_bet = 0
        self.board = 0
        self.deck = create_deck()
        self.fisher_yates_shuffle()

    cpdef reset(self):
        self.deck = create_deck()
        self.fisher_yates_shuffle()

        self.pot = 0
        self.current_bet = 0
        self.board = 0
        for player in self.players:
            player.reset()

    cpdef clone(self):
        cdef GameState new_state = GameState(self.players[:], self.small_blind, self.big_blind)
        new_state.dealer_position = self.dealer_position
        new_state.pot = self.pot
        new_state.current_bet = self.current_bet
        new_state.board = self.board
        new_state.deck = self.deck[:]
        return new_state

    cpdef bint is_terminal(self):
        cdef int active_players = 0
        #cdef Player player
        #cdef int num_active_players = sum([1 for player in self.players if (not player.folded and player.chips > 0)])
        for player in self.players:
            if not player.folded and player.chips > 0:
                active_players += 1
            if active_players > 1:
                break
        
        # Terminal case: all but one player have folded
        if active_players == 1:
            return True

        # Terminal case: all 5 table cards have been dealt and players have acted
        if self.board_has_five_cards():
            cdef bint all_acted = True
            for player in self.players:
                if not player.folded and player.contributed_to_pot < self.current_bet:
                    all_acted = False
                    break

            if all_acted:
                return True

        return False
        

    cdef bint board_has_five_cards(self):
        return bin(self.board).count('1') == 5
    
    cdef void fisher_yates_shuffle(self):
        cdef int i, j
        cdef unsigned long long temp
        srand(1)
        for i in range(len(self.deck) - 1, 0, -1):
            j = rand() % (i + 1)
            temp = self.deck[i]
            self.deck[i] = self.deck[j]
            self.deck[j] = temp

    cpdef draw_card(self):
        self.board |= self.deck.pop()

    cpdef deal_cards(self):
        for _ in range(2):
            for player in self.players:
                if player.chips > 0:
                    player.add_card(self.deck.pop())


cpdef unsigned long long card_to_int(str suit, str value):
    cdef unsigned long long one = 1
    cdef int suit_index = SUITS.index(suit)
    cdef int value_index = VALUES.index(value)
    cdef int bit_position = suit_index * 13 + value_index
    return one << bit_position


cpdef public list create_deck():
        cdef list deck = [card_to_int(suit, value) for suit in SUITS for value in VALUES]
        return deck


cpdef str int_to_card(unsigned long long card):
    cdef int bit_position = -1
    while card > 1:
        card >>= 1
        bit_position += 1
    cdef int suit_index = bit_position // 13
    cdef int value_index = bit_position % 13

    return f'{VALUES[value_index]}{SUITS[suit_index]}'

cpdef unsigned long long card_str_to_int(str card_str):
    return card_to_int(card_str[1], card_str[0])

cpdef str format_hand(unsigned long long hand):
    cdef list cards = [int_to_card(card) for card in create_deck() if card & hand]
    return " ".join(cards)

cpdef display_game_state(GameState game_state, int player_index):
    print(f"Player {player_index + 1}: {format_hand(game_state.players[player_index].hand)}")
    print(f"Board: {format_hand(game_state.board)}")
    print(f"Pot: {game_state.pot}")
    print(f"Chips: {game_state.players[player_index].chips}")