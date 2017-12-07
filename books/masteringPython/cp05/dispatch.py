import functools

@functools.singledispatch
def printer(value):
    print('other: %r' % value)

@printer.register(str)
def str_printer(value):
    print(value)

@printer.register(int)
def ini_printer(value):
    printer('int: %d' % value)  # in fact, return to printer(str)

@printer.register(dict)
def dict_printer(value):
    printer('dict:')
    for k, v in sorted(value.items()):
        printer('  key: %r, value: %r' % (k, v))

# some useful note
print(printer.dispatch(str))
print(printer.registry)