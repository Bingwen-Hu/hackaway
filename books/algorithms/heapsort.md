### definiton
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
-When indicate the heap in an array, the index of the leaves are n/2+1, n/2+2
.. n



