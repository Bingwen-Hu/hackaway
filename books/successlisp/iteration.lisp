;;; chapter 5 introducing iteration

(loop 
   (print "loops forever")
   (return)
   (print "Never show"))                ; never execute

(let ((n 0))
  (loop 
     (when (> n 10) (return))
     (print n) (prin1 (* n n))
     (incf n)))

; same as above
(dotimes (n 11 "I am the result")       ;result is optional
  (print n) (prin1 (* n n)))

(let ((s 0))
  (dotimes (n 11 s)
    (print n) 
    (incf s (* n n))))

(dolist (item '(1 2.0 3.1 4 5.4 6 7))
  (format t "~&~D is~:[n't~;~] a perfect square.~%" item (integerp item)))

(dolist (item `(1 foo "hello" 85.4 2/3 ,#'abs))
  (format t "~&~s is a ~A~%" item (type-of item)))



;;; finally DO is tricky but powerful
(do ((which 1 (1+ which))
     (lst '(foo bar baz qux) (rest lst)))
    ((null lst) 'done)
  (format t "~&Item ~D is ~S.~%" which (first lst)))

;;; It's clear now, so time and efforts matters for every and every thing!
