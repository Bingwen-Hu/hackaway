# character
a = 'a'
print(typeof(a))
print(1 + a)


# char is not string, like C
b = 'c'
s = "c"
print(typeof(b))
print(typeof(s))

# string concatenation
n1 = "Mory"
n2 = "Ann"
believe = "$n1 and $n2 will be togather"
print(believe)
yabelie = *(n1, "-", n2)
print(yabelie)

# start from 1 and end to `end`
print(believe[1:end])

# get the length
length(believe)

# find some thing
# deprecated
search(believe, "Ann")

# string in
occursin("Ann", believe)

# repeat
hi3 = "Hi" ^ 3
print(hi3)

# join and split
joins = join([1, 2, 3], "|")
splits = split(joins, "|")

# reverse
Python = "Python is just good"
print(reverse(Python))

#  randstring
using Random
rstring = Random.randstring(20)