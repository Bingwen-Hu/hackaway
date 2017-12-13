# basic example

class MetaSpam(type):
    
    # notice how the __new__ method has the same arguments
    # as the type function we used earlier. 
    def __new__(metaclass, name, bases, namespace):
        name = 'SpamCreateByMeta'
        bases = (int,) + bases
        namespace['eggs'] = 1
        return type.__new__(metaclass, name, bases, namespace)


# regular Spam
class Spam(object):
    pass


print(Spam.__name__)
print(issubclass(Spam, int))
try:
    Spam.eggs
except Exception as e:
    print(e)


# meta Spam
class Spam(object, metaclass=MetaSpam):
    pass

print(Spam.__name__)
print(issubclass(Spam, int))
print(Spam.eggs)

