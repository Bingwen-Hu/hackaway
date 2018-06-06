;; Write a function that counts the number of times an expression
;; occurs anywhere within another expression.

(defun count-anywhere (x lst) 
  (if (null lst)
      0
      (let ((head (first lst)))
	(cond ((listp head) (+ (count-anywhere x head)
			       (count-anywhere x (rest lst))))
	      ((eql head x) (+ 1 (count-anywhere x (rest lst))))
	      (t (count-anywhere x (rest lst)))))))