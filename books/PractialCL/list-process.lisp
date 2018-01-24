;;; basic parts
(cons 1 2)
(cons 1 nil)
(cons 2 (cons 1 nil))
(cons 3 (cons 2 (cons 1 nil)))
(list 1)
(list 1 2)
(list 1 2 3)


;;; functional programming style
(append (list 1 2) (list 3 5))

;;; destructive operations
(nconc (list 1 2 3) (list 2 4))

