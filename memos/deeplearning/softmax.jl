# softmax illustration
function softmax(v)
    max = maximum(v)
    explist = [exp(x - max) for x in v]
    expsum = sum(explist)
    dist = [exp_ / expsum for exp_ in explist]
end

v = rand(5)
dist = softmax(v)

println(dist)
# softmax function: sum of classes is 1.0
println(sum(dist))