import numpy
cimport numpy

cdef class Deck:

    def __init__(self, list suits, list values):
        self.suits = suits
        self.values = values
        self.reset()
        
    cdef void create_deck(self):
        cdef int index = 0
        self.deck = [0] * len(self.suits) * len(self.values)
        cdef str suit, value
        for suit in self.suits:
            for value in self.values:
                self.deck[index] = card_to_int(suit, value)
                index += 1

    cdef void shuffle(self):
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
        new_deck.deck[:] = self.deck
        return new_deck

    cdef void reset(self):
        self.create_deck()
        self.shuffle()