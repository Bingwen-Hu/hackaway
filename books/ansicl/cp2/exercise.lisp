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
