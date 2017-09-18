# coding: utf-8
# chapter 6 Design Pattern

from abc import ABC, abstractmethod
from collections import namedtuple

Customer = namedtuple('Customer', "name fidelity")

class LineItem:

    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity

class Order:

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount

    def __repr__(self):
        fmt = "<Order total: {:.2f} due: {:.2f}>"
        return fmt.format(self.total(), self.due())

class Promotion(ABC):

    @abstractmethod
    def discount(self, order):
        """Return discount as a positive dollar amount"""


class FidelityPromo(Promotion):
    """5% discount for customers with 1000 or more fidelity points"""

    def discount(self, order):
        return order.total() * .05 if order.customer.fidelity >= 1000 else 0

class BulkItemPromo(Promotion):
    """10% discount for each LineItem with 20 or more units"""

    def discount(self, order):
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * 0.1
        return discount

class LargeOrderPromo(Promotion):
    """7% discount for orders with 10 or more distinct items"""

    def discount(self, order):
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * 0.7
        return 0


#==============================================================================
# Function-Oriented Strategy
#==============================================================================
from collections import namedtuple

Customer = namedtuple("Customer", "name fidelity")

### define class LineItem
### define class Order
class Order:

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion(self)
        return self.total() - discount

    def __repr__(self):
        fmt = "<Order total: {:.2f} due: {:.2f}>"
        return fmt.format(self.total(), self.due())
### function replacer

def fidelity_promo(order):
    """5% discount for customers with 1000 or more fidelity points."""
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0

def bulk_item_promo(order):
    """10% discount for each lineitme with 20 or more units"""
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * 0.1
    return discount

def large_order_promo(order):
    """7% discount for orders with 10 or more distinct items"""

    distinct_items = {item.product for item in order.cart}
    if len(distinct_items) >= 10:
        return order.total() * 0.7
    return 0


### select the best promotion function
promos = [fidelity_promo, bulk_item_promo, large_order_promo]

def best_promo(order):
    """select best discount available"""
    return max(promo(order) for promo in promos)









""" Another design pattern call Command
"""

