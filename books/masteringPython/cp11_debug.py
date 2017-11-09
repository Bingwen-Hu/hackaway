# mastering python chapter11 debug
def spam_generator():
	print('a')
	yield 'spam'
	print('b')
	yield 'spam'
	print('c')
	yield 'spam'
	print('d')
	
if __name__ == '__main__':
	g = spam_generator()
	next(g) # until this time, 'a' and 'spam' printed
	next(g)
	
	