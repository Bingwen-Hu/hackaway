def bubbleSort(lst):
    for i in range(len(lst)-1):
        for j in range(len(lst)-1, i, -1):
            if lst[j] < lst[j-1]:
                lst[j-1], lst[j] = lst[j], lst[j-1]
    return lst


if __name__ == '__main__':
    lst = [8, 11, 2, 1, 4, 19, 9, 14, 13, 5, 6, 22]
    lst = bubbleSort(lst)
    print(lst)