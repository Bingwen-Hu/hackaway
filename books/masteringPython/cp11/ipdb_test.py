import ipdb

def spam(eggs):
	print('eggs:', eggs)
	
if __name__ == '__main__':
	ipdb.set_trace()
	for i in range(5):
		spam(i)