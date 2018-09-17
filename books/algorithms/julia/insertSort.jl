function insertSort(lst)
    for i in 2:length(lst)
        j = i - 1
        key = lst[i]
        while j >= 1 && key > lst[j] 
            lst[j+1] = lst[j]
            j -= 1
        end
        lst[j+1] = key
    end
end


lst = [1, 2, 4, 5, 6, 8, 9, 11, 13, 14, 19, 22]
println(lst)
insertSort(lst)
println(lst)