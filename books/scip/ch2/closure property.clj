; 2.2 hierarchical data and closure property


(def one-through-four (list 1 2 3 4))
one-through-four
(cons 1 one-through-four)
(first one-through-four)
(rest one-through-four)

; list operation
(defn list-ref [items n]
  (if (= n 0)
    (first items)
    (list-ref (rest items) (- n 1))))

(list-ref one-through-four 3)

(defn length [items]
  (if (not (seq items))
    0
    (+ 1 (length (rest items)))))

(length one-through-four)
(length (list 1 2 3))
(length (list 1 'a' "db"))

(defn length [items]
  (defn length-iter [items n]
    (if (not (seq items))
      n
      (length-iter (rest items) (+ n 1))))
  (length-iter items 0))

(length one-through-four)

(defn append [lst1 lst2]
  (if (not (seq lst1))
    lst2
    (cons (first lst1) (append (rest lst1) lst2))))

(append one-through-four (list 11 22 35))

(list)
; e2.17 last-pair
(defn last-pair [items]
  (if (empty? (rest items))
    items
    (last-pair (rest items))))

(def names ["Mory", "Jenny", "Ann"])
(def levels (list 1 2 3 4 5))
(nil? (rest (list 1)))
(empty? (rest (list 1)))
(last-pair names)

; e2.18
(defn reverse [items]
  (if (empty? items)
    ()
    (cons (reverse (rest items)) (first items))))

(reverse levels)
(defn reverse [items]
  (defn reverse-iter [items result]
    (if (empty? items)
        result
        (reverse-iter (rest items) (cons (first items)  result))))
  (reverse-iter items ()))

(reverse levels)
(map (fn [x] (+ 1 x)) levels)


; mapping over lists
(defn mapcar [proc items]
  (if (empty? items)
    nil
    (cons (proc (first items))
          (mapcar proc (rest items)))))

(mapcar (fn [x] (* 2 x))
        levels)

; e2.21
(defn square-list [items]
  (map (fn [x] (* x x))
       items))
(square-list levels)

(defn square-list [items]
  (defn square-list-iter [items result]
    (if (empty? items)
      result
      (square-list-iter (rest items) (cons (square (first items))
                                           result))))
  (reverse (square-list-iter items ())))

(square-list levels)

