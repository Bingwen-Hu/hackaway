;;; In short, transducers are used to saved memory and speed up!

;;; not transducers, imtermediate results are produced!
(declare label-heavy)
(declare non-food?)
(declare unbundle-pallet)

(def process-bags
  (comp 
   (partial map label-heavy)
   (partial filter non-food?)
   (partial mapcat unbundle-pallet)))

;;; using transducers
(def process-bags-fast
  (comp 
   (mapcat unbundle-pallet)
   (filter non-food?)
   (map label-heavy)))

;;; transducers in action
(def xf (map inc))
(transduce xf conj [0 1 2])
(transduce xf conj () [0 1 2])

(def xf (comp 
         (map inc)
         (filter even?)))

(transduce xf conj (range 10))
