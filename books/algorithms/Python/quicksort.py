import random


def partition(lst, p, r, is_random=True):
    if is_random:
        r_ = random.randint(p, r-1)
        lst[r_], lst[r-1] = lst[r-1], lst[r_]
    # print(lst)
    x = lst[r-1]
    i = p

    for j in range(p, r-1):
        if lst[j] <= x:
            lst[j], lst[i] = lst[i], lst[j]
            i += 1
            # print(lst)
    lst[i], lst[r-1] = lst[r-1], lst[i]
    # print(lst)
    return i

def quicksort(lst, p, r, is_random=True):
    if p < r:
        q = partition(lst, p, r)
        quicksort(lst, p, q)
        quicksort(lst, q+1, r)

if __name__ == '__main__':
    lst = [8, 11, 2, 1, 4, 19, 9, 14, 13, 5, 6, 22]
    # i = partition(lst, 0, len(lst))
    quicksort(lst, 0, len(lst))
    print(lst)