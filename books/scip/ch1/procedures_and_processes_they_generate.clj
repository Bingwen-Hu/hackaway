; 1.2.1 linear recursion and iteration

; linear recursion process
; grows and shrinks
(defn factorial [n]
  (if (= n 1)
    1
    (* n (factorial (- n 1)))))

(factorial 5)


; linear iteration process
; no growing and shrinking at all
(defn factorial [n]
  (fact-iter 1 1 n))

(defn fact-iter [product counter max-count]
  (if (> counter max-count)
      product
      (fact-iter (* counter product)
                 (+ counter 1)
                 max-count)))

(factorial 5)

; e1.10 Ackermann's function
(defn A [x y]
  (cond 
    (= y 0) 0
    (= x 0) (* 2 y)
    (= y 1) 2
    :else (A (- x 1)
            (A x (- y 1)))))

(A 1 10)
(A 2 4)
(A 3 3)

(defn f [n]
  (A 0 n))

; f(n) = 2n
(f 4)
(f 1)

(defn g [n]
  (A 1 n))

; g(n) = 2^n
(g 1)
(g 2)
(g 3)
(g 4)
(g 5)

(defn h [n]
  (A 2 n))

(h 0)
(h 1)
(h 2)
(h 3)
(h 4)
; (h 5) ; too large


; 1.2.2 tree recursion
(defn fib [n]
  (cond 
    (= n 0) 0
    (= n 1) 1
    :else (+ (fib (- n 1))
             (fib (- n 2)))))

(fib 8)

; linear iteration version
(defn fib [n]
  (fib-iter 1 0 n))

(defn fib-iter [a b count]
  (if (= count 0)
      b
      (fib-iter (+ a b) a (- count 1))))

(fib 8)

; example: change count
; there is amount and n kinds of coins
; first, pick up one coin, then amount descrease, the total count is become sum of two parts:
; count of the rest amount into n kinds of coins +
; 0

(defn count-change [amount]
  (cc amount 5))

(defn cc [amount kinds-of-coins]
  (cond 
    (= amount 0) 1
    (or (< amount 0) (= kinds-of-coins 0)) 0
    :else (+ (cc amount
                 (- kinds-of-coins 1))
             (cc (- amount 
                    (first-denomination kinds-of-coins))
                 kinds-of-coins)))) 

(defn first-denomination [kinds-of-coins]
  (cond 
    (= kinds-of-coins 1) 1
    (= kinds-of-coins 2) 5
    (= kinds-of-coins 3) 10
    (= kinds-of-coins 4) 25
    (= kinds-of-coins 5) 50))
 
(count-change 100)
(count-change 10)

; e1.11  f(n) = n if n<3 and f(n) = f(n - 1) + 2f(n - 2) + 3f(n - 3) if n> 3. 
; recursive process
(defn f [n]
  (if (< n 3)
    n
    (+ (f (- n 1))
       (* (f (- n 2)) 2)
       (* (f (- n 3)) 3))))

(f 4)
(f 12)
; linear iteration version
(defn f [n]
  (f-iter 2 1 0 n))

(defn f-iter [a b c count]
  (if (= count 0)
      c
      (f-iter (+ a (* 2 b) (* 3 c)) a b (- count 1))))
(f 4)
(f 12)

; e1.12 pascal' triangle
(defn pascal-triangle [n]
  (cond 
    (= 1 n) 1
    (= 2 n) [1 1]
    :else "remain to implement if I still living and remember"))
  
; 1.2.3 order of growth
; 1.2.4 exponentiation
(defn expt [b n]
  (if (= n 0)
      1
      (* b (expt b (- n 1)))))
(expt 2 3)

(defn expt [b n]
  (expt-iter b n 1))
(defn expt-iter [b counter product]
  (if (= counter 0)
      product 
      (expt-iter b (- counter 1) (* b product))))

(expt 2 3)

(defn fast-expt [b n]
  (cond 
    (= n 0) 1
    (even? n) (square (fast-expt b (/ n 2))) ; square in element_of_prog.clj
    :else (* b (fast-expt b (- n 1)))))

(defn even? [n]
  (= (rem n 2) 0))
  
(fast-expt 2 3)
(fast-expt 4 2)

; e1.16
(defn fast-expt [b n]
  (fast-expt-helper b n 1))

(defn fast-expt-helper [b n a]
  "b: base n: expt number a: result
  a * b^n is an invariant quantity"
  (cond 
    (= n 0) a
    (even? n) (fast-expt-helper b (- n 2) (* a b b)) ; square in element_of_prog.clj
    :else (fast-expt-helper b (- n 1) (* a b))))

(fast-expt 2 3)
(fast-expt 4 2)

; e1.17 define a fast multiple operation
(defn fast* [a b]
  "a is the one to repeat
  b is the repeat time"  
  (cond 
    (= 0 b) 0
    (even? b) (mory-double (fast* a (halve b)))
    :else (+ a (fast* a (- b 1)))))

(defn halve [n]
  (/ n 2))

(defn mory-double [n] ; double is a keyword in clojure
  (+ n n))

(mory-double 2)

(halve 3)
(fast* 3 4)

; 1.2.5 Greatest Common Divisors
; equation: GCD(a, b) = GCD(b, r) while r = (rem a b)
(defn gcd [a b]
  "assume a > b"
  (if (= b 0)
      a
      (gcd b (rem a b))))

(gcd 4 6)
(gcd 12 20)
(gcd 100 25)

; 1.2.6 Example: testing for primality
(defn smallest-divisor [n]
  (find-divisor n 2))

(defn find-divisor [n test-divisor]
  (cond
    (> (square test-divisor) n) n
    (divides? test-divisor n) test-divisor
    :else (find-divisor n (+ 1 test-divisor)))) 

(defn divides? [a b]
  (= (rem b a) 0))

(smallest-divisor 239)

(defn prime? [n]
  (= (smallest-divisor n) n))

(prime? 239)
(prime? 123)
(smallest-divisor 199)
(smallest-divisor 1999)
(smallest-divisor 19999)

; Fermat’s Little Theorem: If n is a prime number and a is any positive integer less than n, then a
; raised to the nth power is congruent to a modulo n.
; (rem a^n n) == a
(defn expmod [base exp m]
  (cond 
    (= exp 0) 1
    (even? exp) (rem (square (expmod base (/ exp 2) m))
                     m)
    :else (rem (* base (expmod base (- exp 1) m))
               m)))
  
(defn fermat-test [n]
  (defn try-it [a]
    (= (expmod a n n) a))
  (try-it (+ 1 (rand-int (- n 1)))))

(defn fast-prime? [n times]
  (cond 
    (= times 0) true
    (fermat-test n) (fast-prime? n (- times 1))
    :else false))

(fast-prime? 19 3)

