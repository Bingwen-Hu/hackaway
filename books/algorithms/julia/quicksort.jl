function randint(start, end_)
    r = rand() * (end_ - start)
    r = floor(r) + start
    r = Int(r)
end

function swap(lst, i, j)
    t = lst[i]
    lst[i] = lst[j]
    lst[j] = t
end


function partition(lst, p, r, random=false)
    if random == true
        r_ = randint(p, r+1)
        swap(lst, r, r_)
    end
    x = lst[r]
    i = p
    for j in p:r-1
        if lst[j] <= x
            swap(lst, j, i)
            i += 1
        end
    end
    swap(lst, r, i)
    return i
end

lst = [8, 11, 2, 1, 4, 19, 9, 14, 13, 5, 6, 22, ]
len = length(lst)
function quicksort(lst, p, r, random=false)
    if p < r
        q = partition(lst, p, r, random)
        println(q)
        quicksort(lst, p, q-1)
        quicksort(lst, q+1, r)
    end
end

