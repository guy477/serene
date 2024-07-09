# cython: language_level=3
import numpy
cimport numpy
cimport cython

from libc.stdlib cimport RAND_MAX

import logging
import time

class PokerHandLogger:
    def __init__(self, log_file):
        self.logger = logging.getLogger('PokerHandLogger')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.hand_count = 0

    def log_hand(self, hand_data):
        self.hand_count += 1
        self.logger.info(hand_data)

    def generate_hand_log(self, hand_id, game_type, limit, date_time, dealer_position, seats, blinds, hole_cards, actions, board, showdown, summary):
        hand_log = []
        hand_log.append(f"Hand #{hand_id} - {game_type} ({limit}) - {date_time} UTC")
        hand_log.append(f"Firestone 6-max Seat #{dealer_position} is the button")
        for seat, (player, chips) in enumerate(seats):
            hand_log.append(f"Seat {seat + 1}: {player} (${chips:.2f})")
        for blind, (player, amount) in blinds.items():
            hand_log.append(f"{player} posts the {blind} ${amount:.2f}")
        hand_log.append("*** HOLE CARDS ***")
        for player, cards in hole_cards.items():
            if cards:
                hand_log.append(f"Dealt to {player} [{cards[0]} {cards[1]}]")
            else:
                hand_log.append(f"Dealt to {player} [XX XX]")
        for round_name, round_actions in actions.items():
            if round_actions:
                hand_log.append(f"*** {round_name.upper()} ***")
                for action in round_actions:
                    hand_log.append(f"{action}")
        if board:
            hand_log.append(f"Board {' '.join(board)}")
        if showdown:
            hand_log.append("*** SHOW DOWN ***")
            for player, cards, description in showdown:
                if cards:
                    hand_log.append(f"{player} shows [{cards[0]} {cards[1]}] ({description})")
                else:
                    hand_log.append(f"{player} shows [XX XX] ({description})")
        hand_log.append("*** SUMMARY ***")
        for player, result in summary.items():
            hand_log.append(result)
        return '\n'.join(hand_log)

poker_logger = PokerHandLogger("poker_hand_history.log")

cdef class GameState:
    def __init__(self, list[Player] players, int small_blind, int big_blind, int num_simulations, bint silent=False, list suits=SUITS, list values=VALUES):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.num_simulations = num_simulations
        self.suits = suits
        self.values = values
        self.silent = silent
        self.hand_id = 0
        self.deck = Deck(self.suits, self.values)  # Initialize the Deck object
        self.positions = self.generate_positions(len(self.players))
        self.betting_history = [[], [], [], []]
        self.assign_positions()
        self.reset()


    cpdef void setup_preflop(self, object hand = None):
        self.reset()

        if hand:
            # Remove hand from game_state deck
            self.deck.remove(card_str_to_int(hand[0]))
            self.deck.remove(card_str_to_int(hand[1]))

        self.handle_blinds()

        self.deal_private_cards()
    

    cdef void handle_blinds(self):
        cdef int small_blind_pos, big_blind_pos, small_blind_amount, big_blind_amount
        cdef Player small_blind_player, big_blind_player

        # Calculate small blind and big blind positions
        small_blind_pos = (self.dealer_position + 1) % len(self.players)
        big_blind_pos = (self.dealer_position + 2) % len(self.players)

        # Get the small blind and big blind players
        small_blind_player = self.players[small_blind_pos]
        big_blind_player = self.players[big_blind_pos]

        # Calculate small blind and big blind amounts
        small_blind_amount = min(self.small_blind, small_blind_player.chips)
        big_blind_amount = min(self.big_blind, big_blind_player.chips)

        # Take actions for small blind and big blind
        small_blind_player.take_action(self, ("blinds", small_blind_amount))
        big_blind_player.take_action(self, ("blinds", big_blind_amount))


    cdef void deal_private_cards(self):
        cdef unsigned long long i, card1, card2
        cdef Player player

        # Deal cards to all players
        for i in range(len(self.players)):
            player = self.players[i]
            
            if player.chips > 0:
                card1 = self.deck.pop()
                card2 = self.deck.pop()
                player.add_card(card1)
                player.add_card(card2)
                player.abstracted_hand = abstract_hand(card1, card2)
            else:
                player.folded = True




    cpdef void setup_postflop(self, str round_name):
        # if self.is_terminal_river():
        #     return

        self.cur_round_index += 1
        if round_name == "flop":
            for _ in range(3):
                self.draw_card()
        else:
            self.draw_card()
        self.current_bet = 0
        for player in self.players:
            player.contributed_to_pot = 0
        self.round_active_players = self.active_players()
        self.player_index = (self.dealer_position + 1) % len(self.players)
        self.last_raiser = -1
        self.num_actions = 0

    cdef bint handle_action(self, object action):
        if self.is_terminal() or self.is_terminal_river():
            return True

        # if self.get_current_player().folded or self.get_current_player().chips == 0:

        #     self.player_index = (self.player_index + 1) % len(self.players)
        #     return self.is_terminal()


        if self.get_current_player().take_action(self, action):
            self.last_raiser = self.player_index

        self.num_actions += 1
        self.player_index = (self.player_index + 1) % len(self.players)

        if self.active_players() == self.allin_players():
            self.progress_to_showdown()
            return True

        return self.is_terminal()
    
    cpdef bint step(self, action):
        if self.handle_action(action):
            if self.num_board_cards() == 0:
                self.setup_postflop('flop')
            elif self.cur_round_index < 3: # 0-pre; 1-flop; 2-turn; 3-river.
                self.setup_postflop('postflop')

            return True
        return False


    cdef void showdown(self):
        cdef unsigned long long player_hand, board_dupe
        cdef int best_score, player_score
        cdef int remaining_players = sum(1 for player in self.players if not player.folded)
        cdef list hands = [player.hand for player in self.players]  # Preallocate list of hands
        cdef Player winner

        if self.winner_index != -1:
            return  # Avoid redundant processing

        if remaining_players == 1:
            for i, player in enumerate(self.players):
                if not player.folded:
                    self.winner_index = i
                    break
        else:
            # board_dupe = self.board
            # deck_dupe = self.deck.clone()

            # win_rate = [0] * len(self.players)
            # for _ in range(self.num_simulations):
                # self.board = board_dupe
                # self.deck = deck_dupe.clone()
                # self.deck.fisher_yates_shuffle()

                # while self.num_board_cards() < 5:
                #     self.draw_card()

                # best_score = -1
                # self.winner_index = -1

                

                # for i, player in enumerate(self.players):
                #     player.hand = hands[i]  # Restore hands

                # win_rate[self.winner_index] += 1
            for i, player in enumerate(self.players):
                if player.folded:
                    continue

                if player.hand == 0:
                    player.hand = self.deck.pop() | self.deck.pop()

                player_hand = player.hand | self.board
                player_score = cy_evaluate(player_hand, 7)

                if player_score > best_score:
                    best_score = player_score
                    self.winner_index = i
            # self.winner_index = win_rate.index(max(win_rate))
            num_simulations = float(self.num_simulations)

            # for i, player in enumerate(self.players):
            #     player.expected_hand_strength = win_rate[i] / num_simulations

        winner = self.players[self.winner_index]
        winner.prior_gains += self.pot
        winner.chips += self.pot

        self.log_current_hand(terminal=True)  # Log the hand here if terminal at river
    
#############################################

    cdef void progress_to_showdown(self):
        ## Progress to a terminal state
        # Case: progress_to_showdown called during CFR Training at a depth limit - with the prior action being a raise.
        # TLDR: Get showdown utility to a true terminal state.
        action = ('call', 0) if self.cur_round_index != 0 else ('call', 0)
        while not self.step(action):
            continue
        
        self.cur_round_index = 4

        ## Deal out remaining cards
        # NOTE: Showdown function automatically simulates drawing cards.. but we need to validate it works.
        while self.num_board_cards() < 5:
            self.draw_card()

        
        self.showdown()

    cpdef bint is_terminal(self):
        if (((self.num_actions >= self.round_active_players) and (self.last_raiser == -1 or self.last_raiser == self.player_index)) or
            (self.active_players() == 1 or self.allin_players() == self.active_players())):
            return True
        return False

    cpdef bint is_terminal_river(self):
        if (self.cur_round_index >= 4 or
            (self.board_has_five_cards() and self.is_terminal()) or
            self.active_players() == 1 or self.allin_players() == self.active_players()):
            return True
        return False

    cpdef Player get_current_player(self):
        return self.players[self.player_index]

    cdef int allin_players(self):
        cdef int allin = 0
        for player in self.players:
            if player.chips == 0 and not player.folded:
                allin += 1
        return allin

    cdef int active_players(self):
        cdef int alive = 0
        for player in self.players:
            if not player.folded:
                alive += 1
        return alive

    cdef int folded_players(self):
        cdef int folded = 0
        for player in self.players:
            if player.folded:
                folded += 1
        return folded

    cdef void draw_card(self):
        self.board |= self.deck.pop()

    cdef bint board_has_five_cards(self):
        return self.num_board_cards() == 5

    cdef int num_board_cards(self):
        cdef int count = 0
        cdef unsigned long long temp_board = self.board
        while temp_board:
            count += temp_board & 1
            temp_board >>= 1
        return count

    cdef void assign_positions(self):
        for i in range(len(self.players)):
            self.players[i].assign_position(self.positions[i], i)


#############################################


    cdef void update_current_hand(self, object hand):
        cdef unsigned long long card1, card2
        cdef Player player

        # extract the current player's hand
        player_hand = hand_to_cards(self.get_current_player().hand)

        # Erase the player's hand
        self.get_current_player().hand = 0

        # Add the current player's hand back to the deck
        self.deck.add(player_hand[0])
        self.deck.add(player_hand[1])

        # Convert string card to unsigned int card
        card1 = card_str_to_int(hand[0])
        card2 = card_str_to_int(hand[1])

        ## Remove custom hand from deck
        ## NOTE: Since we pre-emptively remove the desired cards from the deck, we don't need to remove them from the player's hand.
        ##       May require a refactor later on.
        # self.deck.remove(card1)
        # self.deck.remove(card2)

        # Update the player's hand with new hand
        self.get_current_player().add_card(card1)
        self.get_current_player().add_card(card2)
        self.get_current_player().abstracted_hand = abstract_hand(card1, card2)


    cdef void reset(self):
        self.cur_round_index = 0
        self.dealer_position = 0
        self.player_index = (self.dealer_position + 3) % len(self.players)
        self.round_active_players = len(self.players)
        self.num_actions = 0
        self.last_raiser = -1
        self.current_bet = 0
        self.winner_index = -1
        self.pot = 0
        self.board = 0
        self.betting_history[0] = []
        self.betting_history[1] = []
        self.betting_history[2] = []
        self.betting_history[3] = []
        self.deck.reset()
        for i in range(len(self.players)):
            self.players[i].reset()
        
            

    cpdef GameState clone(self):
        # Create a new GameState instance
        cdef GameState new_state = GameState([Player(i.chips, i.bet_sizing, i.is_human) for i in self.players], self.small_blind, self.big_blind, self.num_simulations, self.silent, self.suits, self.values)
        new_state.players = [self.players[i].clone() for i in range(len(self.players))]
        new_state.cur_round_index = self.cur_round_index
        new_state.dealer_position = self.dealer_position
        new_state.player_index = self.player_index
        new_state.num_actions = self.num_actions
        new_state.last_raiser = self.last_raiser
        new_state.pot = self.pot
        new_state.current_bet = self.current_bet
        new_state.board = self.board
        new_state.deck = self.deck.clone()
        new_state.round_active_players = self.round_active_players

        ### NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
        ### Shuffing the deck between clones have drastic down stream affects. un comment if you know what you're doing
        ## With a loose interpretation of "change node"... this will makes every node very chance-ie!
        # new_state.deck.fisher_yates_shuffle()
        
        new_state.betting_history[0][:] = self.betting_history[0]
        new_state.betting_history[1][:] = self.betting_history[1]
        new_state.betting_history[2][:] = self.betting_history[2]
        new_state.betting_history[3][:] = self.betting_history[3]
        new_state.round_active_players = self.round_active_players
        new_state.winner_index = self.winner_index
        new_state.hand_id = self.hand_id

        return new_state

#############################################

    cpdef void debug_output(self):
        print(f"is_terminal(): {self.is_terminal()}")
        print(f"is_terminal_river(): {self.is_terminal_river()}")
        print(f"cur_round_index: {self.cur_round_index}")
        print(f"cur_pot: {self.pot}")
        print(f"cur_bet: {self.current_bet}")
        print(f"num_actions: {self.num_actions}")
        print(f"last_raiser: {self.last_raiser}")
        print(f"player_index: {self.player_index}")
        print(f"dealer_position: {self.dealer_position}")
        print(f"round_active_players: {self.round_active_players}")
        print(f"active_players(): {self.active_players()}")
        print(f"allin_players(): {self.allin_players()}")
        print(f"folded_players(): {self.folded_players()}")
        print(f"current_bet: {self.current_bet}")
        print(f"betting_history: {self.betting_history}")
        print(f"BOARD: {format_hand(self.board)}")
        print(f"deck: {[int_to_card(card) for card in self.deck.to_list()]}")
        print(f"___________")
        for player in self.players:
            print(f"Player {player.player_index}")
            print(f"Player {player.player_index} position: {player.position}")
            print(f"Player {player.player_index} chips: {player.chips}")
            print(f"Player {player.player_index} folded: {player.folded}")
            print(f"Player {player.player_index} abstracted: {player.abstracted_hand}")
            print(f"Player {player.player_index} numeric hand: {format_hand(player.hand)}")
            print(f"Player {player.player_index} contributed: {player.contributed_to_pot}")
            print(f"Player {player.player_index} total contributed: {player.tot_contributed_to_pot}")
            print("")

            
    cdef void log_current_hand(self, terminal=False):
        if self.silent:
            return
        
        seats = [(player.position, player.chips) for player in self.players]
        blinds = {
            "small blind": (self.players[(self.dealer_position + 1) % len(self.players)].position, self.small_blind),
            "big blind": (self.players[(self.dealer_position + 2) % len(self.players)].position, self.big_blind)
        }
        hole_cards = {player.position: [int_to_card(card) for card in hand_to_cards(player.hand)] for player in self.players if not player.folded}
        actions = {
            "pre-flop": self.betting_history[0],
            "flop": self.betting_history[1],
            "turn": self.betting_history[2],
            "river": self.betting_history[3]
        }
        board = [int_to_card(card) for card in hand_to_cards(self.board)]
        showdown = [(player.position, [int_to_card(card) for card in hand_to_cards(player.hand)], 'player.hand_description()') for player in self.players if not player.folded]
        summary = {player.position: 'player.result_description()' for player in self.players}
        hand_log = poker_logger.generate_hand_log(self.hand_id, "Holdem", "No Limit", time.strftime('%Y/%m/%d %H:%M:%S'), self.dealer_position + 1, seats, blinds, hole_cards, actions, board, showdown, summary)
        
        if terminal:
            poker_logger.log_hand(hand_log)
        
        self.hand_id += 1
                    
    cdef list generate_positions(self, int num_players):
        if num_players == 2:
            return ['D', 'SB']
        elif num_players == 3:
            return ['D', 'SB', 'BB']
        else:
            positions = ['D', 'SB', 'BB', 'UTG']
            if num_players > 4:
                positions.append('MP')
            if num_players > 5:
                positions.append('CO')
            if num_players == 7:
                positions.append('MP2')
            elif num_players == 8:
                positions.append('HJ')
            elif num_players == 9:
                positions.append('HJ')
                positions.append('MP2')
            return positions
    
