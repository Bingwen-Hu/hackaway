# cp14 iterables, iterators, and Generators

# Every generator is an iterator: generators fully implement the
# iterator interface.

"""
Every collection in Python is iterable, and iterators are used
internally to support:
• for loops
• Collection types construction and extension
• Looping over text files line by line
• List, dict, and set comprehensions
• Tuple unpacking
• Unpacking actual parameters with * in function calls
"""

import re
import reprlib

RE_WORD = re.compile('\w+')

class Sentence:

    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)

#    def __getitem__(self, index):
#        return self.words[index]
#
#    def __len__(self):
#        return len(self.words)
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)

#    def __iter__(self):
#        return SentenceIterator(self.words)
    def __iter__(self):
        for word in self.words: # 生成器
            yield word
        return


s1 = Sentence("I hope after my heart training, I can get in the state of Buddha")
it = iter(s1)
print(next(it))
print(next(it))



class SentenceIterator:

    def __init__(self, words):
        self.words = words
        self.index = 0

    def __next__(self):
        try:
            word = self.words[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return word

    def __iter__(self):
        return self





#==============================================================================
# A lazy sentence
#==============================================================================


RE_WORD = re.compile('\w+')

class Sentence:

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)

    # generator
    def __iter__(self):
        for match in RE_WORD.finditer(self.text):
            yield match.group()

    # or replace with a genexp
    # def __iter__(self):
    #    return (match.group() for match in RE_WORD.finditer(self.text))

#==============================================================================
# Example: Arithmetic Progression Generator 算术级数
#==============================================================================

class ArithmeticProgression:

    def __init__(self, begin, step, end=None):
        self.begin = begin
        self.step = step
        self.end = end # None -> "infinite" series

    def __iter__(self):
        result = type(self.begin + self.step)(self.begin)
        forever = self.end is None
        index = 0
        while forever or result < self.end:
            yield result
            index += 1
            result = self.begin + self.step * index

# using a generator function
def aritprog_gen(begin, step, end=None):
    result = type(begin + step)(begin)
    forever = end is None
    index = 0
    while forever or result < end:
        yield result
        index += 1
        result = begin + step * index


# using generator tools
import itertools

def aritprog_gen2(begin, step, end=None):
    first = type(begin + step)(begin)
    ap_gen = itertools.count(first, step)
    if end is not None:
        ap_gen = itertools.takewhile(lambda n: n < end, ap_gen)
    return ap_gen
