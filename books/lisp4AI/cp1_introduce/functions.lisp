;;; key notes
; using null to predicate whether a list is nil


(defun mory-and-ann () 
  "Mory and Ann are very important friend."
  "Mory and Ann are not docstring!")
(print (mory-and-ann))

(last '(mory And ann always are in lighted))
(last '(mory And ann always are in lighted) 2)

(defun add-1 (value)
  (+ value 1))

(print (mapcar #'add-1 '(1 2 3 4 5))) 


(print (append '(1 2 3) '(4 5 6)))
;;; different in apply and append
(apply #'append '((1 2 3) (1 2 3)))
(funcall #'append '(1 2 3) '(1 2 3))
(print (append '(1 2 3) 1))


;;; if after mapping, return value is a list of list
;;; and we expect it to be a list, then define mappend
(defun self-and-double (x)
  `(,x ,(+ x x)))

(mapcar #'self-and-double '(1 2 3))

(defun mappend (fn lst)
  (apply #'append (mapcar fn lst)))

(mappend #'self-and-double '(1 2 3))

(defun number-and-negative (x)
  `(,x ,(- x)))

(mappend #'number-and-negative '(1 2 3))

(defun mappend-2 (fn lst)
  (if (null lst)
      nil
      (append (funcall fn (first lst))
              (mappend-2 fn (rest lst)))))


;;; very simple function can be replaced with lambda
(mappend-2 (lambda (x) `(,x ,(* 3 x))) '(1 2 3))
