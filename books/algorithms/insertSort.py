def insertSort(A):
    length = len(A)
    for j in range(1, length):
        key = A[j]
        # insert A[j] into the sorted sequence A[0..j-1]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i = i - 1
        A[i+1] = key

        
if '__main__' == __name__:
    lst = [4, 5, 6, 9, 8, 7]
    print(lst)
    insertSort(lst)
    print(lst)

