;; element of programming
;; three mechanisms to form complex ideas
;; primitive expression
;; means of combination
;; means of abstraction

(+ 1 1)
(+ 137 439)
(/ 10 5)
(+ 2.7 10)
(+ 21 24 14 9)
(+ (* 3
      (+ (* 2 4)
         (+ 3 5)))
   (+ (- 10 7)
      6))

(def size 2)
size
(* 4 size)

(def pi 3.14159)
(def radius 10)
(* pi (* radius radius))
(def circumference (* 2 pi radius))
circumference

; 1.1.4 compound procedures
(defn square [x] (* x x))
(square 4)
(square (+ 2 5))
(square (square 3))

(defn sum-of-square [x y]
  (+ (square x) (square y)))
(sum-of-square 3 4)

; two evaluation method
; fully expand and the reduce is known as normal-order
; evaluate the arguments and then apply is known as applicative-order


;; 1.1.6 conditional expressions and predicates
(defn abs [x]
  (cond 
    (> x 0) x
    (< x 0) (- x)
    :else x))
(abs -5)
(abs 1)
(abs 0.0)

(defn abs [x]
  (if (< x 0) 
    (- x) 
    x))
(abs -8)
(abs 8)
 
(defn >= [x y]
  (or (== x y)
      (> x y)))

(>= 5 3)
(>= 3 11)

; 1.2 translate into prefix form
(/ (+ 5 4 
      (- 2 (- 3 (+ 6 1/3))))
   (* 3 
      (- 6 2)
      (- 2 7)))

; 1.3 sum of two larger numbers of three
(defn sum-of-two-larger [x y z]
  "ugly implement"
  (if (> x y)
    (+ x 
       (if (> y z) y z))
    (if (> x z)
      (+ y x)
      (+ y z))))

(sum-of-two-larger 12 45 21)


