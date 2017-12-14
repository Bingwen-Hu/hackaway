# custom type checks
import abc

class CustomList(abc.ABC):
    'This class implements a list-like interface'
    pass

CustomList.register(list)

# second test
class UniversalClass(abc.ABC):
    @classmethod
    def __subclasshook__(self, subclass):
        return True



if __name__ == '__main__':
    print('first test, one way relationship')
    print(issubclass(list, CustomList))
    print(isinstance([], CustomList))
    print(issubclass(CustomList, list))
    print(isinstance(CustomList(), list))

    print('second test')
    print(issubclass(list, UniversalClass))
    print(isinstance([], UniversalClass))
    print(issubclass(bool, UniversalClass))
    print(isinstance(True, UniversalClass))
    print(issubclass(UniversalClass, bool))
    print(issubclass(UniversalClass, list))