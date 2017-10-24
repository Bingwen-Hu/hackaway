;; mapcar, mapc, mapcan diff from how they return value
(mapcar #'atom (list 1 '(2) "foo" nil))
(mapcar #'+ (list 2 3) (list 1 2 4))

;; MAPC for side effect
(mapc #'(lambda (x y) (print (* x y))) (list 1 0 2) (list 1 4 5))

;; MAPCAN destruct the list
(mapcan #'list (list 1 2 3) (list 1 5 7))

;; maplist
(maplist #'list (list 1 2 3) '(4 5 6))

;; mapl just for side effect
(mapl #'(lambda (x y) (print (append x y))) (list 1 0 2) (list 3 4 5))

;; mapcon
(mapcon #'list (list 1 2 3) (list 4 5 6))


;; MAP and MAP-INTO can work on sequence
;; MAP-INTO is the destructive version of MAP
;; A sequence is either a list or a vector
(map nil #'+ (list 1 2 3) (list 4 5 6))
(MAP 'list #'+ (list 1 2 3) (list 4 5 6))
(map 'vector #'+ (list 1 2 3) (list 4 5 6))
(map '(vector number 3) #'+ (list 1 2 3) (list 34 4 5))

(let ((a (make-sequence 'list 3 :initial-element 0)))
  (print a)
  (map-into a #'+ (list 1 2 3) (list 4 5 6))
  a)



;; mapping functions for filtering

;; take a look at append
(append '(1) nil '(3) '(4))

(defun filter-even-numbers (numbers)
  (mapcan #'(lambda (n)
	      (when (evenp n)
		(list n)))
	  numbers))


(defun filter-evenly-divisible (numerators denominators)
  (mapcan #'(lambda (n d)
	      (if (zerop (mod n d))
		  (list (list n d))
		  nil))
	  numerators denominators))

(some #'(lambda (n) (or (< n 0) 
			(> n 100)))
      (list 0 1 99 100))

(every #'(lambda (w) (>= (length w) 5)) 
       (list "bears" "bulls" "racoon"))

;; reduce combines sequence elements
(reduce #'+ (list 1 2 3 4 5))
(reduce #'* '(1 2 3 4 5))