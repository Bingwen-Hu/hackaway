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
    
