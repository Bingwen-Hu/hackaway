; 1.3 formulating abstractions with higher-order procedures
(defn cube [x]
  (* x x x))

(cube 2)


; 1.3.1 procedures as arguments
; 
; sum[i for i in range(a, b+1)]
(defn sum-integers [a b]
  (if (> a b)
    0
    (+ a (sum-integers (+ a 1) b))))

(sum-integers 1 4)

(defn sum-cubes [a b]
  (if (> a b)
    0
    (+ (cube a) (sum-cubes (+ a 1) b))))

(sum-cubes 1 4)

; converges to pi/8
(defn pi-sum [a b]
  (if (> a b)
    0
    (+ (/ 1.0 (* a (+ a 2)))
       (pi-sum (+ a 4) b))))

(* 8 (pi-sum 1 1000))

; extract the sum pattern from the code above
(defn sum [term a next b]
  (if (> a b)
    0
    (+ (term a)
       (sum term (next a) next b))))

(defn inc [n] (+ n 1))
(defn sum-cubes [a b]
  (sum cube a inc b))
(sum-cubes 1 4)

(defn identity [x] x)
(defn sum-integers [a b]
  (sum identity a inc b))
(sum-integers 1 8)

(defn pi-sum [a b]
  (defn pi-term [x]
    (/ 1.0 (* x (+ x 2))))
  (defn pi-next [x]
    (+ x 4))
  (sum pi-term a pi-next b))

(* 8 (pi-sum 1 1000))

(defn integral [f a b dx]
  (defn add-dx [x] (+ x dx))
  (* (sum f (+ a (/ dx 2.0)) add-dx b)
     dx))

(integral cube 0 1 0.01)

; e1.30 change the recursive implement to linear iterate
(defn sum-linear [term a next b]
  (defn iter [a result]
    (if (> a b)
      result
      (iter (next a) (+ result (term a)))))
  (iter a 0))
(defn sum-integers [a b]
  (sum-linear identity a inc b))
(sum-integers 1 8)
  

; 1.3.2 constructing procedures using lambda 
((fn [x] (+ x 4)) 6)
(defn integral [f a b dx]
  (* (sum f
       (+ a (/ dx 2.0))
       (fn [x] (+ x dx))
       b)
     dx))

(integral cube 0 1 0.01)

; local variable
(let [x 2
      f cube]
  (f x))

(let [f (fn [x] (+ x 1))
      x 4]
  (f x))



; 1.3.3 procedures as general methods
; finding roots of equations by the half-interval method
(defn search [f neg-point pos-point]
  (let [midpoint (average neg-point pos-point)]
    (if (close-enough? neg-point pos-point)
      midpoint
      (let [test-value (f midpoint)]
        (cond 
          (pos? test-value) (search f neg-point midpoint)
          (neg? test-value) (search f midpoint pos-point)
          :else midpoint)))))

(defn close-enough? [x y]
  (< (abs (- x y)) 0.001))

(defn half-interval-method [f a b]
  (let [a-value (f a)
        b-value (f b)]
    (cond 
      (and (negative? a-value) (positive? b-value)) 
      (search f a b)
      (and (negative? b-value) (positive? a-value))
      (search f b a)
      :else (println "Values are not of opposite sign"))))

(defn myf [x]  (+ 14 (* -3 x)))
(myf 1)
(myf 10)
(half-interval-method myf -1 10.0)
(myf 4.66)

; fixed point: f(x) = x
; using f(x), f(f(x)), f(f(f(x))) to approximate...

; 1.4 procedure as return values
(defn average-damp [f]
  (fn [x] (average x (f x))))

