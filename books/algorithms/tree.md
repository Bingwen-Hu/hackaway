### binary search tree
Given x is a node in a bi-search-tree, if y is a node in x.left, then
y.val<=x.val; if y is a node in x.right, then y.val >= x.val

### tree functions

    Inorder-Tree-Walk(tree)
    1 if x != NULL
    2      Inorder-Tree-Walk(tree.left)
    3      print(tree.val)
    4      Inorder-Tree-Walk(tree.right)

    preorder and postorder are just similar except that preorder print the
    value of the root first and postorder print it last


    Tree-Search(tree, k)
    1 if x == NIL or k == x.val
    2     return x
    3 if k < x.val
    4     return Tree-Search(tree.left, k)
    5 return Tree-Search(tree.right, k)


    Tree-Minimum(tree)
    1 if tree.left == NIL
    2     return tree.val
    3 return Tree-Minimum(tree.left)

    tree-maximum is very similar -- just always return the right child


    Tree-Successor(tree)
    1 if tree.right != NULL
    2     return Tree-Minimum(tree.right)
    3 y = x.p
    4 while y != NULL and x == y.right
    5     x = y
    6     y = y.p
    7 return y

    the successor s of the node x means:
    s.val > x.val and s.val = min[value larger than x]
    When the tree.right != NULL is easy. but if not?
    the successor is the first ancestor whose left tree contains x


    Tree-Insert(tree, z)
     1 y = NIL
     2 x = tree
     3 while x != NIL
     4     y = x
     5     if z.val < x.val
     6          x = x.left
     7     else x = x.right
     8 x.p = y
     9 if y == NIL
    10     tree = z
    11 elseif z.val < y.val
    12     y.left = z
    13 else y.right = z

    Delete a node is a little complex and we first define a helper function

    Transplant(tree, u, v)
    "using v to replace u totally"
    1 if u.p == NIL
    2      tree = v
    3 elseif u == u.p.left
    4      u.p.left = v
    5 else u.p.right = v
    6 if v != NIL
    7      v.p = u.p

    Tree-Delete(tree, z)
     1 if z.left == NIL
     2     Transplant(tree, z, z.right)
     3 elseif z.right == NIL
     4     Transplant(tree, z, z.left)
     5 else y = tree-minimum(z.right)        // find the successor y
     6     if y.p != z
     7          Transplant(tree, y, y.right) // remove y from the tree
     8          y.right = z.right            // save what z left
     9          y.right.p = y                // complement with line8
    10     Transplant(tree, z, y)
    11     y.left = z.left
    12     y.left.p = y
    
### Red Black Tree
A RB-tree is a binary search tree but every node has an additional property
name color, which is whether RED or BLACK.

Five attribute:
1. Any node is whether Red or Black
2. Root node is Black
3. Leaf node is Black
4. If one node is Red, then its child should be black
5. For every node, to simple road from itself to its every descendant contants
   the same number of black nodes.

### functions

    Left-Rotate(T, x)
     1 y = right                // set y 
     2 x.right = y.left		// turn y's left subtree into x's right subtree
     3 if y.left != T.nil
     4      y.left.p = x
     5 y.p = x.p		// link x's parent to y
     6 if x.p == T.nil
     7      T.root = y
     8 elseif x == x.p.left
     9      x.p.left = y
    10 else x.p.right = y
    11 y.left = x		// put x on y's left
    12 x.p = y


    Right-Rotate(T, y)
     1 x = y.left
     2 y.left = x.right
     3 if x.right != T.nil
     4      x.right.p = y
     5 x.p = y.p
     6 if y.p == T.nil
     7      T.root = x
     8 elseif y == y.p.left
     9      y.p.left = x
    10 else y.p.right = x
    11 y.p = x
    12 x.right = y


    RB-Insert(T, z)
     1 y = T.nil
     2 x = T.root
     3 while x != T.nil
     4     y = x
     5     if z.val < x.val
     6          x = x.left
     7     else x = x.right // so x hold the place and y is parent
     8 z.p = y
     9 if y == T.nil
    10     T.root = z
    11 elseif z.val < y.val
    12     y.left = z
    13 else y.right = z
    14 z.left = T.nil
    15 z.right = T.nil
    16 z.color = RED
    17 RB-Insert-Fixup(T, z)

    RB-Insert-Fixup(T, z)
     1 while z.p.color == RED
     2     if z.p == z.p.p.left
     3          y = z.p.p.right
     4          if y.color == RED
     5               x.p.color = BLACK
     6               y.color = BLACK
     7               z.p.p.color = RED
     8               z = z.p.p
     9          else if z == z.p.right  // y.color == BLACK
    10               z = z.p            // after line10 and line11
    11               Left-Rotate(T, z)  // case2 turn to case3
    12          else z.p.color = BLACK
    13               z.p.p.color = RED
    14               Right-Rotate(T, z)
    15     else (same as then clause with 'right' and 'left' exchanged)
    16 T.root.color = BLACK

    RB-Transplant(T, u, v)
    1 if u.p == T.nil
    2    T.root = v
    3 elseif u == u.p.left
    4    u,p.left = v
    5 else u.p.right = v
    6 v.p = u.p

    RB-Delete(T, z)
     1 y = z
     2 y-original-color = y.color
     3 if z.left == T.nil
     4    x = z.right
     5    RB-Transplant(T, z, z.right)
     6 elseif z.right == T.nil
     7    x = z.left
     8    RB-Transplant(T, z, z.left)
     9 else y = Tree-Minimum(z.right)
    10    y-original-color = y.color
    11    x = y.right
    12    if y.p == x
    13        x.p = y
    14    else RB-Transplant(T, y, y.right)
    15        y.right = z.right
    16        y.right.p = y
    17    RB-Transplant(T, z, y)
    18    y.left = z.left
    19    y.left.p = y
    20    y.color = z.color
    21 if y-original-color == BLACK
    22    RB-Delete-Fixup(T, x)
     
    RB-Delete-Fixup(T,x)
     1 while x != T.root and x.color == BLACK
     2    if x == x.p.left
     3         w = x.p.right
     4         if w.color == RED
     5              w.color = BLACK
     6              x.p.color = RED
     7              Left-Rotate(T, x.p)
     8              w = x.p.right
     9         if w.left.color == BLACK and w.right.color == BLACK
    10              w.color = RED
    11              x = x.p
    12         elseif w.right.color == BLACK
    13              w.left.color = BLACK
    14              w.color = RED
    15              Right-Rotate(T, w)
    16              w = x.p.right
    17         else w.color = x.p.color
    18              x.p.color = BLACK
    19              w.right.color = BLACK
    20              Left-Rotate(T, x.p)
    21              x = T.root
    22    else (same as then clause with 'right' and 'left' exchanged)
    23 x.color = BLACK
