### definition
Heap is a complete binary tree, that is, all levels of the tree, except
possibly the last one are fully filled, and, if the last level of the tree is
not complete, the nodes of that level are filled from left to right.
Heap have two property (I found it confusing)

-length: length of the array

-heapsize: number of elements in the heap

-we have: 0 <= heapsize <= length


### basic function
Given index i, we have:

      Parent(i)
      1 return floor(i/2) // in Clang, i >>= 1

      Left(i)
      1 return 2i // in Clang, i <<= 1

      Right(i)
      1 return 2i+1 // in Clang, i <<= 1; i++

### max-heap min-heap and height 
max-heap: heaps where the parent key is greater than or equal to the child
      A[Parent[i]] >= A[i] except the root 

min-heap: heaps where the parent key is less than or equal to the child
      A[Parent[i]] <= A[i] except the root

height: the number of line from the node to the leave, we call it is the
height of the node. So, the height of the heap is the height of the root

### some notes:

-Given height h, the elements n is in range of [power(2, h), power(2, h+1)-1]

-The height of a n-elements heap is floor(log2(n))

-Given a n-heap in array the index of the leaves are n/2+1, n/2+1, .. n

### useful functions

    Max-heapify(A, i)
    "set the value of A[i] to a proper position"
     1 l = Left(i)
     2 r = Right(i)
     3 if i<=A.heapsize and A[l] >= A[i]
     4      largest = l
     5 else largest = i
     6 if r<=A.heapsize and A[r] >= A[i]
     7      largest = r
     8 if largest != i
     9      exchange(A[i], A[largest])
    10      Max-heapify(A, largest)


    Build-max-heap(A)
    "build the heap from button to top"
    1 A.heapsize = A.length
    2 for i = A.length/2 downto 1
    3      Max-heapify(A, i)

    Notes: the range of [A.length/2, 1] => nodes except the deepest level

    Heapsort(A)
    1 Build-max-heap(A)
    2 for i = A.length downto 2
    3      exchange(A[i], A[1])
    4      A.heapsize -= 1
    5      Max-heapify(A, 1)

    Notes: everytime move the largest value to the last position

### priority queue -- using binary heap struction

    Heap-maximum(A)
    1 return A[i]

    Heap-extract-max(A)
    1 if A.heapsize < 1
    2     error "heap overflow"
    3 max = A[i]
    4 A[1] = A[A.heapsize]
    5 A.heapsize -= 1
    6 Max-heapify(A, 1)
    7 return max

    Heap-increase-key(A, i, key)
    1 if key < A[i]
    2     error "new key is smaller than current key"
    3 A[i] = key
    4 while i > 1 and A[Parent(i)] < A[i]
    5     exchange(A[i], A[Parent(i)])
    6     i = Parent(i)

    Max-heap-insert(A, key)
    1 A.heapsize += 1
    2 A[A.heapsize] = -MAX_INT
    3 Heap-increase-key(A, A.heapsize, key)
