; 2.1 introduction to data abstraction
; 2.1.1 arithmetic operations for rational numbers

; wishful thinking
(defn make-rat [n p] [n p])
(defn numer [r] (first r))
(defn denom [r] (second r))

(defn add-rat [x y]
  (make-rat (+ (* (numer x) (denom y))
               (* (numer y) (denom x)))
            (* (denom x) (denom y))))

(defn sub-rat [x y]
  (make-rat (- (* (numer x) (denom y))
               (* (numer y) (denom x)))
            (* (denom x) (denom y))))

(defn mul-rat [x y]
  (make-rat (* (numer x) (numer y))
            (* (denom x) (denom y))))

(defn div-rat [x y]
  (make-rat (* (numer x) (denom y))
            (* (denom x) (numer y))))

(defn equal-rat? [x y]
  (= (* (numer x) (denom y))
     (* (denom x) (numer y))))

(defn print-rat [r]
  (let [s (str (numer r) "/" (denom r))]
    (println s)))

(def rat1 (make-rat 2 4))
(def rat2 (make-rat 1 3))
(numer rat1)
(denom rat1)
(equal-rat? rat1 rat2)
(print-rat (add-rat rat1 rat2))

; using gcd
(defn make-rat [n d]
  (let [g (gcd n d)]
    (vector (/ n g) (/ d g))))

(make-rat -3 3)

; e2.1 normalize sign
; in fact, gcd just do this thing!
(defn make-rat [n d]
  (let [g (gcd n d)]
    (cond 
      (or (and (neg? n) (neg? d))
          (and (pos? n) (neg? d)))
      (vector (/ (- n) g) (/ (- d) g))
      :else 
      (vector (/ n g) (/ d g)))))



; e2.2
(defn make-segment [x y]
  (vector x y))

(defn start-segment [segment]
  (first segment))

(defn end-segment [segment]
  (second segment))

(defn make-point [x y]
  (vector x y))

(defn x-point [p]
  (first p))

(defn y-point [p]
  (second p))

(defn midpoint-segment [segment]
  (let [start-p (start-segment segment)
        end-p (end-segment segment)]
    (make-point (average (x-point start-p)
                         (x-point end-p))
                (average (y-point start-p)
                         (y-point end-p)))))
(def seg1 (make-segment (make-point 1 5)
                        (make-point -4 2)))
(midpoint-segment seg1)

; e2.3
(defn make-rectangle [left-top right-bottom])
  

; 2.1.3 what is meant by data?

; a interesting example -- you never know what he is, 
; you know what he appears!
; Buddha!
(defn cons-mory [x y]
  (defn dispatch [m]
    (cond 
      (= m 0) x
      (= m 1) y
      :else (str "Argument not 0 or 1 -- cons" m)))
  dispatch)
(defn car [z]
  (z 0))
(defn cdr [z]
  (z 1))

(def x (cons-mory "A" 2))
(car x)
(cdr x)

; The point of exhibiting the procedural representation of pairs is not that our language works this way
; (Scheme, and Lisp systems in general, implement pairs directly, for efficiency reasons) but that it
; could work this way. The procedural representation, although obscure, is a perfectly adequate way to
; #represent pairs, since it fulfills the only conditions that pairs need to fulfill. This example also
; demonstrates that the ability to manipulate procedures as objects automatically provides the ability to
; represent compound data. This may seem a curiosity now, but procedural representations of data will
; play a central role in our programming repertoire. This style of programming is often called message
; passing, and we will be using it as a basic tool in chapter 3 when we address the issues of modeling
; and simulation. 

; e2.4 
(defn cons-ann [x y]
  (fn [m] (m x y)))
(defn car [z]
  (z (fn [p q] p)))
