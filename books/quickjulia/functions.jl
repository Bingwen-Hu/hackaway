function fib(n)
    if n < 0
        return -1
    elseif n <= 2
        return 1
    else
        return fib(n-1) + fib(n-2)
    end
end


for i in 1:10
    println(fib(i))
end

# short hand
fib_short(n) = fib(n)


function length_vec(x, y, z)
    length = sqrt(x^2 + y^2 + z^2)
end

# position parameter

# lambda
result = map(x -> x * 2, [1, 2, 3])
result2 = map(sin, [0, pi])
result3 = map(//, 1:10, 2:11)
# time it!
@time map(sin, 1:10000)

# reduce
reduce(+, 1:10)
foldl(-, 1:3)
foldr(-, 1:3)


# multiple dispatch
f(x, y) = x + y
print(f(1, 2))