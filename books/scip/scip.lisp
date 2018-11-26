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

;;; 1.8 note: buggy
(defun average-cube (x y)
  (/ (+ (/ x (square y))
        (* 2 y))
     3))

(defun mycube (x)
  (cube-iter 1.0 x))

(defun cube-improve (guess x)
  (average-cube x guess))

(defun cube-iter (guess x)
  (if (cube-good-enough? guess x)
      guess
      (cube-iter (cube-improve x guess) x)))

(defun cube-good-enough? (guess x)
  (let ((next (cube-improve x guess)))
    (< (myabs (- next guess))
       0.001)))

(mycube 2)
