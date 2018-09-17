lst = [2, 3, 8, 1, 4, 9, 14, 13, 5, 6]
len = length(lst)

function randint(start, end_)
    r = rand() * (end_ - start)
    r = floor(r) + start
    r = Int(r)
end

function partition(lst)
    
end