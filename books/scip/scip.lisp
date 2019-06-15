;; Common lisp come back

(+ 1 1)
(+ 137 439)
(/ 10 5)
(+ 2.5 10)
(+ 21 24 14 0)
(+ (* 3
      (+ (* 2 4)
         (+ 3 5)))
   (+ (- 10 5)
      7))


(setq size 2)
(* 4 size)

; (defparameter pi 3.14159) special constant
(setq radius 10)
(* pi (* radius radius))

(setq circumference (* 2 pi radius))


(defun square (x) (* x x))
(square 4)

(defun sum-of-squares (x y)
  (+ (square x) 
     (square y)))

(sum-of-squares 5 6)


; 1.1.6 conditional expressions and predicates
(defun myabs (x)
  (cond ((> x 0) x)
        ((= x 0) 0)
        ((< x 0) (- x))))

(myabs -1)
(myabs 1)
(myabs 0)

(defun myabs (x)
  (cond ((< x 0) (- x))
        (t x)))

(myabs -1)
(myabs 1)
(myabs 0)


(defun my>=  (x y)
  (or (> x y) (= x y)))

(my>= 3 5)


;;; 1.2
(/ (+ 5 4 (- 2 (- 3 (+ 6 1/3))))
   (* 3 (- 6 2) (- 2 7)))


;;; 1.3 sum squares of two larger numbers
(defun sum-squares-of-two-larger (x y z)
  (cond  ((and (< x y) (< x z)) 
          (sum-of-squares y z))
         ((and (< y x) (< y z))
          (sum-of-squares x z))
         (t (sum-of-squares x y))))

(sum-squares-of-two-larger 1 2 3)


;;; 1.4 
(defun a-plus-abs-b (a b)
  (funcall (if (> b 0) #'+ #'-) 
           a b))


(defun sqrt-iter (guess x)
  (if (good-enough? guess x)
      guess
      (sqrt-iter (improve guess x)
                 x)))

(defun good-enough? (guess x)
  (< (myabs (- (square guess) x)) 0.001))

(defun improve (guess x)
  (average guess (/ x guess)))

(defun average (x y)
  (/ (+ x y) 2))


(defun mysqrt (x)
  (sqrt-iter 1.0 x))

(mysqrt 9)
(square (mysqrt 1000))

;;; 1.7
(defun good-enough? (guess x)
  (let ((next (improve guess x)))
    (< (myabs (- next guess))
       0.001)))
(square (mysqrt 1000))

;;; 1.8 note: buggy fixed
(defun average-cube (guess x)
  (/ (+ (/ x (square guess))
        (* 2 guess))
     3))

(defun mycube-root (x)
  (cube-root-iter 1.0 x))

(defun cube-improve (guess x)
  (average-cube guess x))

(defun cube-root-iter (guess x)
  (if (cube-good-enough? guess x)
      guess
      (cube-root-iter (cube-improve guess x) 
                      x)))

(defun cube-good-enough? (guess x)
  (let ((next (cube-improve guess x)))
    (< (myabs (- next guess))
       0.00001)))

(mycube-root 2)

; A procedure definition should be able to suppress details

;; introduce block structure and lexical scoping
(defun mysqrt (x)
  (defun good-enough? (guess)
    (< (myabs (- (square guess) x)) 0.001))
  (defun improve (guess)
    (average guess (/ x guess)))
  (defun sqrt-iter (guess)
    (if (good-enough? guess)
        guess
        (sqrt-iter (improve guess))))
  (sqrt-iter 1.0))


                                              
; Great! Here is 1.2 Procedures and the Processes they Generate
; We are still lack of typical openings, tactics, or strategy
; The ability to visualize the consequences of the actions under consideration is crucial to becoming an expert programmer

; 1.2.1 Linear Recursion and Iteration

; using trace can see the difference bwtween the following two functions
(defun factorial (n)
  "recursive process"
  (if (= n 1)
      1
      (* n (factorial (- n 1)))))

(defun factorial (n)
  "iterative recursion"
  (labels ((helper (n ret)
             (if (= n 1)
                 ret
                 (helper (1- n) (* ret n)))))
    (helper n 1)))

;;; 1.2.2 Tree Recursion

(defun fib (n)
  (cond ((= n 0) 0)
        ((= n 1) 1)
        (t (+ (fib (- n 1))
              (fib (- n 2))))))

(defun fib (n)
  "iteration process"
  (defun fib-iter (a b count)
    (if (= count 0)
        b
        (fib-iter (+ a b) a (1- count))))
  (fib-iter 1 0 n))

;;; Example: Counting Change
;;; $ 100 to 5, 2, 1

; the mory version
(defun count-change (amount coins)
  (cond ((= amount 0) 1)
        ((or (< amount 0) (= (length coins) 0)) 0)
        (t (+ (count-change (- amount (first coins))
                            coins)
              (count-change amount (rest coins))))))

;;; 1.12 pascal's triangle
(defun pascal-triangle (n)
  (labels ((pascal-tri (row col)
             (cond ((or (= col 1) (= row col)) 1)
                   (t (+ (pascal-tri (1- row) (1- col))
                         (pascal-tri (1- row) col))))))
    (loop for i from 1 to n
         collect (pascal-tri n i))))
