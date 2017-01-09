# quickselect is a selection algorithm to find the ith smallest element in
# an unordered list. It was developed by Tony Hoare, who developed the
# quicksort as well

# there is also a partition

def swap(data, index1, index2):
    data[index1], data[index2] = data[index2], data[index1]

def partition(data, left, right, pivotIndex):
    "data is the unordered list"
    pivotValue = data[pivotIndex]
    swap(data, pivotIndex, right)
    storeIndex = left
    for i in range(left, right):
        if data[i] < pivotValue:
            swap(data, i, storeIndex)
            storeIndex += 1
    swap(data, right, storeIndex)
    return storeIndex

""" Returns the k-th smallest element of list within left..right inclusive
(i.e. left <= k <= right).
The search space within the array is changing for each round - but the list
is still the same size. Thus, k does not need to be updated with each round. 
"""
def select(data, left, right, k):
    if left == right:
        return data[left]
    pivotIndex = (left + right) // 2
    pivotIndex = partition(data, left, right, pivotIndex)
    # =========== debug code =================
    print("debug code: ", data)
    # ========================================
    if k == pivotIndex:
        return data[k]
    elif k < pivotIndex:
        return select(data, left, pivotIndex-1, k)
    else:
        return select(data, pivotIndex+1, right, k)


if __name__ == "__main__":
    data = [1,4,9,87,35,48,43,76,30,28,19]
    k = 4
    left = 0
    right = len(data) -1
    print("the original data: ")
    print(data)

    result = select(data, left, right, k)
    print("the %d smallest number: %d" % (k, result))
    print("the partial sorted data:")
    print(data)
