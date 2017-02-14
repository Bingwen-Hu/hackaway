# merge sort is the base divide-conquer algorithms 

def merge(A, p, q, r):
    """A: the array store all the data, 
    p: the low index, r: the high index,
    q: the index divide the array.
    """
    
    # copy the left and right array
    left = A[p:q]               # new list 
    right = A[q:r+1]            # right is not A[q:r+1]
    
    length = len(A)
    llen = len(left)
    rlen = len(right)
    
    i = j = 0
    
    for k in range(length):
        if i == llen or j == rlen: # one of arrays is empty
            break

        if left[i] <= right[j]:
            A[k] = left[i]
            i += 1
        else:
            A[k] = right[j]
            j += 1

    
    if i == llen:
        A[i+j:] = right[j:]
    elif j == rlen:
        A[i+j:] = left[i:]

    return A

# the main function is wrong
# something about in-place modification
# waiting for future or forever
def mergeSort(A, p, r):
    if p < r:
        q = (p + r) // 2
        mergeSort(A, p, q)
        mergeSort(A, q+1, r)
        A = merge(A, p, q, r)
        print(A)
    return A
