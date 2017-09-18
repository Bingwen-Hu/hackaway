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

import abc

class Tombola(abc.ABC):

    @abc.abstractmethod
    def load(self, iterable):
        """Add items from an iterable"""

    @abc.abstractmethod
    def pick(self):
        """Remove item at random, returning it.

        This method should raise "LookupError" when the instance is empty
        """

    def loaded(self):
        """Return `True` if there's at least 1 item, `False` otherwise."""
        return bool(self.inspect())

    def inspect(self):
        """Return "a sorted tuple with the items currently inside."""
        items = []
        while True:
            try:
                items.append(self.pick())
            except LookupError:
                break
        self.load(items)
        return tuple(sorted(items))


# subclassing the Tombola ABC
import random

class BingoCage(Tombola):

    def __init__(self, items):
        self._randomizer = random.SystemRandom()
        self._items = []
        self.load(items)

    def load(self, items):
        self._items.extend(items)
        self._randomizer.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')

    def __call__(self):
        self.pick()


class LotteryBlower(Tombola):

    def __init__(self, iterable):
        self._balls = list(iterable)

    def load(self, iterable):
        self._balls.extend(iterable)

    def pick(self):
        try:
            position = random.randrange(len(self._balls))
        except ValueError:
            raise LookupError('pick from empty BingoCage')
        return self._balls.pop[position]

    def loaded(self):
        return bool(self._balls)

    def inspect(self):
        return tuple(sorted(self._balls))


# virtual subclass of tombola: will not inherit any methods or attributes from the
from random import randrange

@Tombola.register
class TomboList(list):

    def pick(self):
        if self:
            position = randrange(len(self))
            return self.pop(position)
        else:
            raise LookupError('pop from empty TomboList')

    load = list.extend


    def loaded(self):
        return bool(self)

    def inspect(self):
        return tuple(sorted(self))



