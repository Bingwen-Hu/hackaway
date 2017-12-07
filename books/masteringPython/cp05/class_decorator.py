import functools

def plus_one(function):
    @functools.wraps(function)
    def _plus_one(self, n):          # self is passed
        return function(self, n + 1)
    return _plus_one


class Spam(object):
    @plus_one
    def get_eggs(self, n=2):
        return n * 'eggs'


####### classmethod and staticmethod
# classmethod passes a class object instead of a class instance
# staticmethod skip both class object and class instance

### how descriptor functions?
class MoreSpam(object):

    def __init__(self, more=1):
        self.more = more

    def __get__(self, instance, owner):
        return self.more + instance.spam

    def __set__(self, instance, value):
        instance.spam = value - self.more

class Spam(object):

    more_spam = MoreSpam(5)

    def __init__(self, spam):
        self.spam = spam


# our own classmethod and staticmethod
import functools

class ClassMethod(object):

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, owner):
        @functools.wraps(self.method)
        def method(*args, **kwargs):
            return self.method(owner, *args, **kwargs)
        return method

class StaticMethod(object):

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, owner):
        return self.method


# property
class Spam(object):

    def get_eggs(self):
        print('getting eggs')
        return self._eggs

    def set_eggs(self, eggs):
        print('setting eggs to %s' % eggs)
        self._eggs = eggs

    def delete_eggs(self):
        print('deleting eggs')
        del self._eggs

    eggs = property(get_eggs, set_eggs, delete_eggs)

    @property
    def spam(self):
        print('getting spam')
        return self._spam

    @spam.setter
    def spam(self, spam):
        print('setting spam to %s' % spam)
        self._spam = spam

    @spam.deleter
    def spam(self):
        print('deleting spam')
        del self._spam

# define our own property
class Property(object):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        # if no specific documentation is available, copy it
        # from the getter
        if fget and not doc:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, instance, owner):
        if instance is None:
            # redirect class (not instance) properties to self
            return self
        elif self.fget:
            return self.fget(instance)
        else:
            raise AttributeError('unreadable attribute')

    def __set__(self, instance, value):
        if self.fset:
            self.fset(instance, value)
        else:
            raise AttributeError("can't set attribute")

    def __delete__(self, instance):
        if self.fdel:
            self.fdel(instance)
        else:
            raise AttributeError("can't delete attribute")

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel)

