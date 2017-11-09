# mastering python chapter11 debug
def eggs_generator():
	yield 'eggs'
	yield 'EGGS'
	
def spam_generator():
	yield 'spam'
	yield 'spam!'
	yield 'SPAM!'
	
generator = spam_generator()
print(next(generator))
print(next(generator))

generator = eggs_generator()
print(next(generator))

# execute:
# python3 -m trace --trace --timing cp11_trace.py