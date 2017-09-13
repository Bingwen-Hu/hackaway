# Chapter 8 Class

# Identity test: using is
charle = {'name': "Charles Dodgson"}
lewis = charle  # alias
print(charle is lewis)
print(id(charle), id(lewis))


# The == operator compares the values of objects (the data they hold),
# while is compares their identities.


# list always change in place but tuple just create a new one.
l1 = [3, [66, 55, 44], (7, 8, 9)]
l2 = list(l1) #
l1.append(100) #
l1[1].remove(55) #
print('l1:', l1)
print('l2:', l2)
l2[1] += [33, 22] #
l2[2] += (10, 11) #
print('l1:', l1)
print('l2:', l2)


# modular copy provide a function to handle deepcopy
class Bus:

    def __init__(self, passengers=None):
        if passengers is None:
            self.passengers = []
        else:
            self.passengers = list(passengers)

    def pick(self, name):
        self.passengers.append(name)

    def drop(self, name):
        self.passengers.remove(name)

# 实际上是，深度复制把像list这种可变容器的底层地址都复制了
# Mutable Types as Parameter Defaults: Bad Idea
import copy
bus1 = Bus(['Ann', "Mory", "Jenny", "Sirius"])
bus2 = copy.copy(bus1)
bus3 = copy.deepcopy(bus1)
print(id(bus1), id(bus2), id(bus3))

bus1.drop('Sirius')
print(bus2.passengers)
print(id(bus2.passengers), id(bus1.passengers), id(bus3.passengers))
print(bus3.passengers)


# Defensive Programming with Mutable Parameters
# 弥合设计者与调用者之间期望的差距

class TwilightBus:
    """会让人消失的暮巴士"""
    def __init__(self, passengers=None):
        if passengers is None:
            self.passengers = []
        else:
            self.passengers = passengers # 直接使用参数

    def pick(self, name):
        self.passengers.append(name)

    def drop(self, name):
        self.passengers.remove(name)

basketball_team = ['Sue', 'Tina', 'Maya', "Diana", "Pat"]
bus = TwilightBus(basketball_team)
bus.drop('Tina')
bus.drop('Pat')
print(basketball_team)


#==============================================================================
# weak reference
# ==============
# Weak references to an object do not increase its reference count. The object that is the
# target of a reference is called the referent. Therefore, we say that a weak reference does
# not prevent the referent from being garbage collected.
# Weak references are useful in caching applications because you don’t want the cached
# objects to be kept alive just because they are referenced by the cache.
#==============================================================================
import weakref
s1 = {1, 2, 3}
s2 = s1
def bye():
    print("Gone with the wind...")

ender = weakref.finalize(s1, bye) # register
print(ender.alive)
del s1
print(ender.alive)
s2 = 'spam'
print(ender.alive)



