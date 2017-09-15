# Inheritance: For Good or For Bad

# We started our coverage of inheritance explaining the problem with subclassing builtin
# types: their native methods implemented in C do not call overridden methods in
# subclasses, except in very few special cases. That’s why, when we need a custom list,
# dict, or str type, it’s easier to subclass UserList, UserDict, or UserString—all defined
# in the collections module, which actually wraps the built-in types and delegate op‐
# erations to them—



class A:
    def ping(self):
        print('ping:', self)

class B:
    def pong(self):
        print("pong:", self)


class C(A):
    def pong(self):
        print('PONG:', self)

class D(B, C):

    def ping(self):
        super().ping()
        print('post-ping:', self)

    def pingpong(self):
        self.ping()
        super().ping()
        self.pong()
        super().pong()
        C.pong(self)


d = D()
d.pong()
C.pong(d)
print(D.__mro__)



# inspect the__mro__ attribute in several classes

def print_mro(cls):
    print(', '.join(c.__name__ for c in cls.__mro__))

#==============================================================================
# Modern Example: Mixins in Django Generic Views
#==============================================================================
# Originally, Django provided a set of functions, called generic views, that implemented
# some common use cases. For example, many sites need to show search results that
# include information from numerous items, with the listing spanning multiple pages,
# and for each item a link to a page with detailed information about it. In Django, a list
# view and a detail view are designed to work together to solve this problem: a list view
# renders search results, and a detail view produces pages for individual items.
