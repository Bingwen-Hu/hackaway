;;; input is list of ints
;;; ints lie in [0, 1, ..., N-1
(defvar N 40)
(defun duplicate (lst)
  (let ((hold (make-array N)))
    (dolist (i lst)
      (incf (aref hold i))
      (if (> (aref hold i) 1)
          (return t)))))

(defun replace-space (str))

(defun search-matrix)


(defun reverse-print (lst)
  "using recursive"
  (if (not (null (rest lst)))
      (reverse-print (rest lst)))
  (print (first lst)))

(defun reverse-print-stack (lst)
  (let ((stack '()))
    (dolist (e lst)
      (push e stack))
    (dolist (e stack)
      (print e))))
