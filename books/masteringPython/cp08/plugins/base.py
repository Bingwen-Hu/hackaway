import os
import re
import abc
import importlib

MODULE_NAME_RE = re.compile('[a-z][a-z0-9_]*', re.IGNORECASE)


class Plugins(abc.ABCMeta):
    plugins = dict()

    def __new__(metaclass, name, bases, namespace):
        cls = abc.ABCMeta.__new__(metaclass, name, bases, namespace)
        if isinstance(cls.name, str):
            metaclass.plugins[cls.name] = cls
        return cls

    @classmethod
    def get(cls, name):
        if name not in cls.plugins:
            print('Loading plugins from plugins.%s' % name)
            importlib.import_module('plugins.%s' % name)
        return cls.plugins[name]

    @classmethod
    def load(cls, *plugin_modules):
        for plugin_module in plugin_modules:
            print('Loading plugins %s' % plugin_module)
            plugin = importlib.import_module(plugin_module)

    @classmethod
    def load_directory(cls, module, directory):
        for file_ in os.listdir(directory):
            name, ext = os.path.splitext(file_)
            full_path = os.path.join(directory, file_)
            import_path = [module]
            if os.path.isdir(full_path):
                import_path.append(file_)
            elif ext == '.py' and MODULE_NAME_RE.match(name):
                import_path.append(name)
            else:
                # ignoring non-matching files/directories
                continue
            plugin = importlib.import_module('.'.join(import_path))


class Plugin(metaclass=Plugins):
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplemented()
