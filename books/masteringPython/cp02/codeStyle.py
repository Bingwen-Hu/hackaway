# Zen of Python
# Style Guide

# Beautiful is better than ugly
# when do very complex thing and listcomp seems unreadable

# not append
def filter_modulo(items, modulo):
    output_items = []
    for i in range(len(items)):
        if items[i] % modulo:
            output_items.append(items[i])
    return output_items


# not unreadable listcomp
filter_modulo = lambda i, m: [i[j] for j in range(len(i))
                              if i[j] % m]


# using generator
def filter_modulo(items, modulo):
    for item in items:
        if item % modulo:
            yield item

# flake8 is a tool Mory you should be familiar with

# issues: modify when iterate, using list to make a copy
set_ = {'spam', 'eggs'}
for item in list(set_):
    set_.remove(item)