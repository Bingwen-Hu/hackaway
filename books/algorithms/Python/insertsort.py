def insertsort(lst):
    for i in range(1, len(lst)):
        key = lst[i]
        j = i - 1
        while j >= 0 and lst[j] < key:
            lst[j+1] = lst[j]
            j -= 1
        lst[j+1] = key
    return lst

if __name__ == '__main__':
    lst = [8, 11, 2, 1, 4, 19, 9, 14, 13, 5, 6, 22]
    lst = insertsort(lst)
    print(lst)