import numpy
cimport numpy

cdef class Deck:

    def __init__(self, list suits, list values):
        self.suits = suits
        self.values = values

    cdef list create_deck(self):
        cdef int size = len(self.suits) * len(self.values)
        cdef list deck = [0] * size
        cdef int index = 0
        cdef int suit_index, value_index
        cdef str suit, value
        for suit in self.suits:
            suit_index = SUITS_INDEX[suit]
            for value in self.values:
                value_index = VALUES_INDEX[value]
                deck[index] = card_to_int(suit, value)
                index += 1
        return deck

    cdef void fisher_yates_shuffle(self):
        numpy.random.shuffle(self.deck)
        # pass

    cdef unsigned long long pop(self):
        # Remove from the beginning
        return self.deck.pop(0)

    cdef void remove(self, unsigned long long card):
        # Remove the given card
        self.deck.remove(card)

    cdef void add(self, unsigned long long card):
        # Add to the end
        self.deck.append(card)

    cdef list to_list(self):
        # Deck is a list; but if we were to optimize deck to another struct...
        return self.deck

    cdef Deck clone(self):
        cdef Deck new_deck = Deck(self.suits, self.values)
        new_deck.deck = self.deck[:]
        return new_deck

    cdef void reset(self):
        self.deck = self.create_deck()
        self.fisher_yates_shuffle()