# Chapter 20
# this chapter is a key to mastering python
# A class implementing a __get__, a __set__, or a __delete__ method is a descriptor.

class Quantity:
    def __init__(self, storage_name):
        self.storage_name = storage_name

    # instance is the managed instance
    def __set__(self, instance, value):
        if value > 0:
            instance.__dict__[self.storage_name] = value
        else:
            raise ValueError('value must be > 0')

class LineItem:
    # Descriptor is a protocol - based feature;no subclassing is needed to implement one.
    weight = Quantity('weight')
    price = Quantity('price')

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price

if __name__ == '__main__':
    truffle = LineItem('White truffle', 100, 0)