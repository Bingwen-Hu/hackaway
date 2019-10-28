import vec

my = vec.My()

# use native list
lst = [1, 2, 3, 4]
my.print_vector(lst)

# use C++ vector<int>
v = vec.vectori([5, 6, 7, 8])
my.print_vector(v)
