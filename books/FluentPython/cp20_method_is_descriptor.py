import collections

class Text(collections.UserString):

    def __repr__(self):
        return 'Text({!r})'.format(self.data)

    def reverse(self):
        return self[::-1]

# Any function is a nonoverriding descriptor. Calling its __get__ with an instance
# retrieves a method bound to that instance.
# Mory example
class Empty:
    pass

def myfun():
    print('good mod!')

method = myfun.__get__(Empty)
print(method)