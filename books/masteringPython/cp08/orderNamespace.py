# make namespace in order
import collections


class Field(object):
    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return '<%s %s>' % (
            self.__class__.__name__,
            self.name,
        )


class FieldMeta(type):
    @classmethod
    def __prepare__(metaclass, name, bases):
        return collections.OrderedDict()

    def __new__(metaclass, name, bases, namespace):
        cls = type.__new__(metaclass, name, bases, namespace)
        cls.fields = []
        for k, v in namespace.items():
            if isinstance(v, Field):
                cls.fields.append(v)
                v.name = v.name or k
        return cls


class Fields(metaclass=FieldMeta):
    spam = Field()
    eggs = Field()


print(Fields.fields)
fields = Fields()
print(fields.fields)
