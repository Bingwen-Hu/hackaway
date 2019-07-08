;;; input is list of ints
;;; ints lie in [0, 1, ..., N-1
(defvar N 40)
(defun duplicate (lst)
  (let ((hold (make-array N)))
    (dolist (i lst)
      (incf (aref hold i))
      (if (> (aref hold i) 1)
          (return t)))))

(defun replace-space (str)
  "ugly implementation in lisp"
  (let ((longstr str))
    (dotimes (s (length str))
      (if (char= (char str s) #\space)
          (setf longstr (concatenate 'string longstr "  "))))
    (let ((p1 (- (length str) 1))
          (p2 (- (length longstr) 1)))
      (dotimes (i (length str))
        (if (char= (char str p1) #\space)
            (progn
              (setf (char longstr p2) #\0)
              (decf p2)
              (setf (char longstr p2) #\2)
              (decf p2)
              (setf (char longstr p2) #\%)
              (decf p2))
            (progn
              (setf (char longstr p2) (char str p1))
              (decf p2)))
        (decf p1)))
    longstr))


(defparameter matrix #2A((1 4 7 11 15)
                         (2 5 8 12 19)
                         (3 6 9 16 22)
                         (10 13 14 17 24)
                         (18 21 23 26 30)))
(defun search-matrix (matrix target)
  (multiple-value-bind (rows cols) (array-dimensions matrix)
    ))


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

(defun rebuild-binary-tree (preorder inorder)
  )
