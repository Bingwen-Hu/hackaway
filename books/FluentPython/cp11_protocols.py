""" Chapter 11 protocols and ABC

duck typing: operating with objects regardless of their types, as long as
they implement certain protocols."""


#==============================================================================
# Monkey-Patching
#==============================================================================
import collections
Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list("JQKA")
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                       for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]


from random import shuffle
l = list(range(10))
shuffle(l)
print(l)

deck = FrenchDeck()
shuffle(deck)

# Monkey patch
def set_card(deck, position, card):
    deck._cards[position] = card

FrenchDeck.__setitem__ = set_card
deck = FrenchDeck()
shuffle(deck)
print(deck)



#==============================================================================
# frenchdeck2: subclassing an ABC
#==============================================================================
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck2(collections.MutableSequence):
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = "spades diamonds clubs hearts".split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                       for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

    def __setitem__(self, position, value):
        self._cards[position] = value

    def __delitem__(self, position):
        del self._cards[position]

    def insert(self, position, value):
        self._cards.insert(position, value)

# Python does not check for the implementation of the abstract methods at import time
# (when the frenchdeck2.py module is loaded and compiled), but only at runtime when
# we actually try to instantiate FrenchDeck2. Then, if we fail to implement any abstract
# method, we get a TypeError exception with a message such as "Can't instantiate
# abstract class FrenchDeck2 with abstract methods __delitem__, insert".
# That’s why we must implement __delitem__ and insert, even if our FrenchDeck2
# examples do not need those behaviors: the MutableSequence ABC demands them.

