# cp19 LineItem, good example to see the protected property in Python

class LineItem:

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight              # property setter is already in use
        self.price = price

    def subtotal(self):
        return self.weight * self.price

    @property
    def weight(self):
        return self.__weight

    @weight.setter
    def weight(self, value):
        if value > 0:
            self.__weight = value
        else:
            raise ValueError('value must be > 0')


