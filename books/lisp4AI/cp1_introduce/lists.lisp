;;; all things in Lisp is List

(defparameter lst '(Mory and Ann meets in a beautiful spring))
(print (length lst))
(print (car lst))
(print (caddr lst))
(format nil "~a and ~a" (car lst) (caddr lst))

