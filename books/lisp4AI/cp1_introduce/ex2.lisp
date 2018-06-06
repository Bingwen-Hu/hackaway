;; Write a function to exponentiate, or raise a number to an integer power
(defun power (num exp) 
  (labels ((power-helper (num exp res)
	     (if (= exp 0)
		 res
		 (power-helper num (- exp 1) (* res num)))))
    (power-helper num exp 1)))