# accessing metaclass attributes through classes

class Meta(type):

    @property
    def spam(cls):
        return 'Spam property of %r' % cls

    def eggs(self):
        return 'Eggs method of %r' % self

class SomeClass(metaclass=Meta):
    pass

if __name__ == '__main__':
    print(SomeClass.spam)
    try:
        print(SomeClass().spam)
    except Exception as e:
        print(e)
    print(SomeClass.eggs())
    try:
        print(SomeClass().eggs())
    except Exception as e:
        print(e)



# Test Note:
# property and method only accessiable by class but not instance
# though the author say it seems useless.