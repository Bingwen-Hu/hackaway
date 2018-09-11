# list comprehensions
sqlist = [sqrt(i) for i in 1:10]

# generators
gen = (x for x in 1:100 if x % 7 == 0)
for i in gen
    println(i)
end

# enumerate
for (index, value) in enumerate(gen)
    println("$index -> $value")
end

# zip
for (a, b) in zip(1:5, 6:10)
    println("$a -> $b")
end