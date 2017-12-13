# two ways to define class

class Spam(object):
    eggs = 'my eggs'

print('classic define')
spam = Spam()
print(spam.eggs)
print(type(spam))
print(type(Spam))



Spam = type('Spam', (object,), dict(eggs='my eggs'))


print('using type')
spam = Spam()
print(spam.eggs)
print(type(spam))
print(type(Spam))
