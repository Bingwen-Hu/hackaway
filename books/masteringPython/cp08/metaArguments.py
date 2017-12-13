# metaclasses with arguments
class MetaWithArguments(type):
    def __init__(metaclass, name, bases, namespace, **kwargs):
        # the kwargs should not be passed on to the type.__init__
        type.__init__(metaclass, name, bases, namespace)

    def __new__(metaclass, name, bases, namespace, **kwargs):
        for k, v in kwargs.items():
            namespace.setdefault(k, v)

        return type.__new__(metaclass, name, bases, namespace)


class WithArgument(metaclass=MetaWithArguments, spam='eggs'):
    pass


if __name__ == '__main__':
    with_argument = WithArgument()
    print(with_argument.spam)

