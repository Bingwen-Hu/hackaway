;;; tree list as a tree
(defparameter tree '((a b) ((c)) (d e)))
(tree-equal tree (copy-tree tree))

(defun same-shape-tree (a b)
  "Are two trees the same except for the leaves?"
  (tree-equal a b :test #'true))

(defun true (&rest ignore) t)
(same-shape-tree tree '((1 2) ((3)) (4 5)))

(same-shape-tree tree '((1 2) (3) (4 5)))

;;; subst in a tree
(let ((tree '(I (Am Mory) and Mory is I)))
  (subst 'Ann 'Mory tree))
(let ((tree '(I (Am Mory) and Mory is I)))
  (sublis '((Mory . Ann))
          tree))

;;; sublis using a mapping

