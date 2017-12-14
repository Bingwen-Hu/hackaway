# emulate the same behavior with metaclass
import functools

class AbstractMeta(type):
    def __new__(metaclass, name, bases, namespace):
        # create the class instance
        cls = super().__new__(metaclass, name, bases, namespace)

        # collect all local methods as abstract
        abstracts = set()
        for k, v in namespace.items():
            if getattr(v, '__abstract__', False):
                abstracts.add(k)

        # look for abstract methods in the base classes and add
        # them to the list of abstracts
        for base in bases:
            for k in getattr(base, '__abstract__', ()):
                v = getattr(cls, k, None)
                if getattr(v, "__abstract__", False):
                    abstracts.add(k)

        # store the abstracts in frozenset so they cannot be modified
        cls.__abstract__ = frozenset(abstracts)

        # decorate the __new__ function to check if all abstract
        # functions were implemented
        original_new = cls.__new__
        @functools.wraps(original_new)
        def new(self, *args, **kwargs):
            for k in self.__abstract__:
                v = getattr(self, k)
                if getattr(v, '__abstract__', False):
                    raise RuntimeError(
                        '%r is not implemented' % k
                    )
            return original_new(self, *args, **kwargs)

        cls.__new__ = new
        return cls


def abstractmethod(function):
    function.__abstract__ = True
    return function

class Spam(metaclass=AbstractMeta):
    @abstractmethod
    def some_method(self):
        pass

if __name__ == '__main__':
    try:
        eggs = Spam()
    except Exception as e:
        print(e)


# Test Note:
# I can not understand....