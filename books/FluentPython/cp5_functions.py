# Chapter 5 first-class functions


#==============================================================================
# function as object
#==============================================================================
def factorial(n):
    """returns n!"""
    return 1 if n < 2 else n * factorial(n-1)

print(factorial.__doc__)
print(type(factorial))

fact = factorial
print(fact(5))
ls = list(map(factorial, range(11)))
print(ls)

#==============================================================================
# high order function
#==============================================================================
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
print(sorted(fruits, key=len))

def reverse(word):
    return word[::-1]
print(sorted(fruits, key=reverse))



#==============================================================================
# Anonymous functions
#==============================================================================
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
print(sorted(fruits, key=lambda word: word[::-1]))


# callable check
print([callable(obj) for obj in (abs, str, 14)])

#==============================================================================
# From Positional to Keyword-Only Parameters
#==============================================================================
def tag(name, *content, cls=None, **attrs):
    """Generate one or more HTML tags"""
    if cls is not None:
        attrs['class'] = cls
    if attrs:
        attr_str = "".join(' %s="%s"' % (attr, value) for
                           attr, value in sorted(attrs.items()))
    else:
        attr_str = ""
    if content:
        return "\n".join('<%s%s>%s</%s>' %
                         (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)
print(tag('br'))
print(tag('p', 'hello'))
print(tag('p', 'hello', id=33))
print(tag('p', 'hello', 'world', cls='sidebar'))

# even "content" is considered as a keyword parameter
print(tag(content="testing", name="img"))
my_tag = {'name': 'img', 'title': 'Sunset Boulevard',
          'source_test': 'sunset.jpg', 'cls': 'framed'}
print(tag(**my_tag))

#==============================================================================
# inspect
#==============================================================================

def clip(text, max_len=80):
    """Return text clipped at the last space before or after max_len
    """
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after

    if end is None: # no spaces were found
         end = len(text)
    return text[:end].rstrip()
print(clip.__defaults__)
print(clip.__code__.co_varnames)
print(clip.__code__.co_argcount)

# a better way
from inspect import signature
sig = signature(clip)
print(str(sig))
for name, param in sig.parameters.items():
    print(param.kind, ':', name, '=', param.default)

#==============================================================================
# Functions annotations
#==============================================================================
def clip2(text:str, max_len:'int > 0'=80) -> str:
    """Return text clipped at the last space before or after max_len
    """
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after

    if end is None: # no spaces were found
         end = len(text)
    return text[:end].rstrip()

print(clip2.__annotations__)

#==============================================================================
# functional modules
#==============================================================================
from functools import reduce, partial
from operator import mul, methodcaller

def fact(n):
    return reduce(mul, range(1, n+1))

s = "The time to the end"
upcase = methodcaller('upper')
print(upcase(s))

triple = partial(mul, 3)
print(triple(7))