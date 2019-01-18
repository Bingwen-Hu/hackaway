; e2 three distinct cons expressions that return (a b c)
'(a b c)
(list 'a 'b 'c)
(cons 'a (cons 'b (cons 'c nil)))


; e3 using car and cdr define a function return the fourth element of a list
(defun my-fourth (lst)
  (car (cdr (cdr (cdr lst)))))

(my-fourth '(1 2 3))
(my-fourth '(1 2 3 4 5))

; e4 Define a function that takes two arguments and returns the greater of the two.
(defun larger-of-two (a b)
  (if (< a b)
      b
      a))

; e5 What do these functions do?

(defun enigma (x)
  (and (not (null x))
       (or (null (car x))
           (enigma (cdr x)))))
; answer: test whether any nil in x

(defun mystery (x y)
  (if (null y)
      nil
      (if (eql (car y) x)
          0
          (let ((z (mystery x (cdr y))))
            (and z (+ z 1))))))
; search the x index in y

; e6 
(car (car (cdr '(a (b c) d))))
(or 13 (/ 1 0))
(funcall #'list 1 nil)
(apply #'list 1 nil)

; e7 takes a list as an argument and returns true if one of its elements is a list
(defun list-as-element (lst)
  (if (null lst)
      nil
      (if (listp (car lst))
          t
          (list-as-element (cdr lst)))))

(list-as-element '(1 2 3))
(list-as-element '(1 (2 3) 4))

