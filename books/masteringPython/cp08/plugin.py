# automatically registering a plugin system
# one of the most common uses of metaclasses is to have classes automatically register
# themselves as plugins/handlers

import abc

class Plugins(abc.ABCMeta):
    plugins = dict()

    def __new__(metaclass, name, bases, namespace):
        cls = abc.ABCMeta.__new__(metaclass, name, bases, namespace)
        if isinstance(cls.name, str):
            metaclass.plugins[cls.name] = cls
            return cls

    @classmethod
    def get(cls, name):
        return cls.plugins[name]

class PluginBase(metaclass=Plugins):
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplemented()

class EggsPlugin(PluginBase):
    name = 'eggs'

class SpamPlugin(PluginBase):
    name = 'spam'


# if __name__ == '__main__':
#     print(Plugins.get('spam'))
#     print(Plugins.plugins)