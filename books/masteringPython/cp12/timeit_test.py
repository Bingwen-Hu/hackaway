# chapter12 performace
# timeit module

import timeit

def test_list():
	return list(range(1000))
	
def test_list_comprehension():
	return [i for i in range(1000)]
	
def test_append():
	x = []
	for i in range(1000):
		x.append(i)
		
	return x
	
def test_insert():
	x = []
	for i in range(10000):
		x.insert(0, i)
		
	return x
	
def benchmark(function, number=100, repeat=10):
	# Measure the execution times
	times = timeit.repeat(function, number=number, globals=globals())
	# the repeat function gives 'repeat' results so we take the min()
	time = min(times) / number
	print('%d loops, best of %d: %9.6fs :: %s' % (
		number, repeat, time, function))
		
if __name__ == '__main__':
	benchmark('test_list()')
	benchmark('test_list_comprehension()')
	benchmark('test_append()')
	benchmark('test_insert()')