function bubbleSort(lst)
    len = length(lst)
    for i in 1:len-1
        j = len
        while j > i
            if lst[j] < lst[j-1]
                lst[j], lst[j-1] = lst[j-1], lst[j]
            end
            j -= 1
        end
    end
    lst
end

lst = [8, 11, 2, 1, 4, 19, 9, 14, 13, 5, 6, 22]
lst = bubbleSort(lst)
println(lst)