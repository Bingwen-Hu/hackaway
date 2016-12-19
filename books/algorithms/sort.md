### Heap
Heap is a complete binary tree, that is, all levels of the tree, except
possibly the last one are fully filled, and, if the last level of the tree is
not complete, the nodes of that level are filled from left to right.
Heap have two property (I found it confusing)


length: length of the array

heapsize: number of elements in the heap

we have: 0 <= heapsize <= length


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

Given height h, the elements n is in range of [power(2, h), power(2, h+1)-1]

The height of a n-elements heap is floor(log2(n))

Given a n-heap in array the index of the leaves are n/2+1, n/2+1, .. n

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
    best time: nlogn
    worst time: nlogn


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


### Quicksort
divide the array into two part A[p..q-1] and A[q+1..r], the choice of q
affects the effection of the sort.
Mean time: nlogn
Worst time: n^2

    Quicksort(A, p, r)
    1 if p < r
    2    q = Partition(A, p, r)
    3    Quicksort(A, p, q-1)
    4    Quicksort(A, q+1, r)

Given a q, it can lead to different partition
the Partition followed use the last number as the q. It is better to choose a
random one.

    Partition(A, p, r)
    1 x = A[r]
    2 i = p -1
    3 for j = p to r-1
    4     if A[j] <= x
    5          i = i + 1
    6          exchange(A[i], A[j])
    7 exchange(A[i+1], A[r])
    8 return i+1

    Randomized-Partition(A, p, r)
    "Randomize the choice of q"
    1 i = Random(p, r)
    2 exchange(A[r], A[i])
    3 return Partition(A, p, r)

    Randomized-Quicksort(A, p, r)
    1 if p < r
    2     q = Randomized-Partition(A, p, r)
    3     Randomized-Quicksort(A, p, q-1)
    4     Randomized-Quicksort(A, q+1, r)

### Counting-Sort
Without using comparition but using the count. This algorithms is typically
using space to trade off time.

    Counting-Sort(A, B, k)
     1 let C[0..k] be a new array
     2 for i = 0 to k
     3     C[i] = 0
     4 for j = 1 to A.length
     5     C[A[j]] = C[A[j]] + 1
     6 // C[i] now contains the number of elements equal to i
     7 for i = 1 to k
     8     C[i] = C[i] + C[i-1]
     9 // C[i] now contains the number of elements less than or equal to i
    10 for j = A.length downto 1
    11     B[C[A[j]]] = A[j]
    12     C[A[j]] = C[A[j]] - 1

the most important trick is that if you know how many elements less than you,
than you know where you should be. If you are equal, note that line 10 -- the
loop is from max to min, so who first in will first out. That is, the counting
sort is steady

### radix and bucket sort I have to find out when I need them.
