;;; demo of loops

; simple loops
(defvar my-vector #(1 0 2 4 3 0))
(defvar my-list '(1 0 2 4 3 6))

;; error on vector
(loop for v in my-vector
     collect v)

;; using across
(loop for val across my-vector
   collect val)

;; on list is ok
(loop for i in my-list
     collect i)

;; loop with index
(loop for ele in my-list
   for i from 1 to (length my-list)
   collect (list i ele))

;; loop with index by step
(loop for ele in my-list
     for i from 1 to 60 by 5
     collect (list i ele))


;; countting backwards
(loop for i from 10 downto -10 by 5
     collect i)


;; loop on
(loop for i on my-list
     collect i)

(loop for x in my-list
   collect x into collected
   count x into counted
   sum x into sumed
   maximize x into maximized
   minimize x into minimized
   finally (return (list collected counted sumed maximized minimized)))


;; loop for conditional
(loop for x in my-list
     when (evenp x) sum x)
