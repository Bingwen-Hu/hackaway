import functools

def singleton(cls):
    instances = dict()
    @functools.wraps(cls)
    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _singleton

@singleton
class Spam(object):
    def __init__(self):
        print('Executing init')


spam = Spam()
print(type(spam))
