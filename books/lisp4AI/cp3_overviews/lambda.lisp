;; exercise 3.1
(let* ((x 6)
       (y (* x x)))
  (+ x y))

;; using lambda to mimic let*
((lambda (x y)
   (+ x y))
 ((lambda (x)
   (* x x))
  6)
 6)

