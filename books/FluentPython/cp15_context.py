# chapter 15 Context managers and else block

# The semantics of for/else, while/else, and try/else are closely related,
# but very different from if/else.

""" Mory Note:
in for, while, try, the else clause can be think as `then`
"""

# some falsey example
for item in my_list:
    if item.flavor == 'banana':
        break
else:
    raise ValueError("No banana flavor found!")




try:
    dangerous_call()
    after_call()
except OSError:
    log("OSError...")

# should be better as following
try:
    dangerous_call()
except OSError:
    log("OSError...")
else:
    after_call()



#==============================================================================
# Context Managers and with Blocks
#==============================================================================
class LookingGlass:

    def __enter__(self):
        import sys
        self.original_write = sys.stdout.write
        sys.stdout.write = self.reverse_write
        return "JABBERWOCKY"


    def reverse_write(self, text):
        self.original_write(text[::-1])

    def __exit__(self, exc_type, exc_value, traceback):
        import sys
        sys.stdout.write = self.original_write
        if exc_type is ZeroDivisionError:
            print('Please DO NOT divide by zero!')
            return True


import contextlib

@contextlib.contextmanager
def looking_glass():
    import sys
    original_write = sys.stdout.write

    def reverse_write(text):
        original_write(text[::-1])

    sys.stdout.write = reverse_write
    msg = ''
    try:
        yield 'JABBERWOCKY'
    except ZeroDivisionError:
        msg = "Please DO NOT divide by zero!"
    finally:
        sys.stdout.write = original_write
        if msg:
            print(msg)


#==============================================================================
# a context manager for rewriting files in place
#==============================================================================
import csv

with inplace(csvfilename, 'r', newline='') as (infh, outfh):
    reader = csv.reader(infh)
    writer = csv.writer(outfh)

    for row in reader:
        row += ['new', 'columns']
        writer.writerow(row)