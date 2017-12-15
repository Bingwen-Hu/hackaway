# the order of operations with instantiating classes
# wait for more reviews

# the order
# __Prepare__: preparing the namespace
# executing the class body
# creating the class object
# executing the class decorators
# creating the class instances


import functools


def decorator(name):
    def _decorator(cls):
        @functools.wraps(cls)
        def __decorator(*args, **kwargs):
            print('decorator(%s)' % name)
            return cls(*args, **kwargs)

        return __decorator

    return _decorator


class SpamMeta(type):

    @decorator("SpamMeta.__init__")
    def __init__(self, name, bases, namespace, **kwargs):
        print('SpamMeta.__init__()')
        return type.__init__(self, name, bases, namespace)

    @staticmethod
    @decorator('SpamMeta.__new__')
    def __new__(cls, name, bases, namespace, **kwargs):
        print("SpamMeta.__new__()")
        return type.__new__(cls, name, bases, namespace)

    @classmethod
    @decorator('SpamMeta.__prepare__')
    def __prepare__(cls, names, bases, **kwargs):
        print('SpamMeta.__prepare__()')
        namespace = dict(spam=5)
        return namespace


@decorator('Spam')
class Spam(metaclass=SpamMeta):

    @decorator('Spam.__init__')
    def __init__(self, eggs=10):
        print('Spam.__init__()')
        self.eggs = eggs


# test with class object
spam = Spam
print(spam)
try:
    spam.eggs
except Exception as e:
    print(e)

# test with class instance
spam = Spam()
print(spam.spam)
print(spam.eggs)
