;; compare an expression with values
(let ((x 10))
  (case x
    ((1 2 3) 'YES)
    ((10 20 30) 'No)
    (otherwise 'Mory)))

(let ((x 'Ann))
  (typecase x
    (Symbol 'Mory)
    (list '(Mory))
    (number 11)
    (otherwise 'Jenny)))

;; ecase and etypecase will cause error if there is no match