# demo of regular abstract base class

import abc

# abstract class, could not create instance
class Spam(object, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def some_property(self):
        raise NotImplemented()

    @classmethod
    @abc.abstractmethod
    def some_classmethod(self):
        raise NotImplemented()

    @staticmethod
    @abc.abstractmethod
    def some_staticmethod():
        raise NotImplemented()

    @abc.abstractmethod
    def some_method(self):
        raise NotImplemented()

class Eggs(Spam):
    def some_new_method(self):
        pass

class Bacon(Spam):
    def some_method(self):
        pass

    @property
    def some_property(self):
        pass

    @classmethod
    def some_classmethod(self):
        pass

    @staticmethod
    def some_staticmethod():
        pass



if __name__ == '__main__':
    # only the class implement abstract method can create instance
    try:
        eggs = Eggs()
    except Exception as e:
        print(e)

    bacon = Bacon()

    try:
        spam = Spam()
    except Exception as e:
        print(e)