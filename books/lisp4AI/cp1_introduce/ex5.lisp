;; Write a function to compute the dot product of two sequences
;; of numbers, represented as lists. The dot product is computed by multiplying
;; corresponding elements and then adding up the resulting products

(defun dot-product (v1 v2)
  (if (or (null v1) (null v2))
      0
      (+ (* (first v1) (first v2))
	 (dot-product (rest v1) (rest v2)))))