(defun arith-eval (expr)
  (cond ((numberp expr) expr)
	(t (funcall (second expr)
		    (arith-eval (first expr))
		    (arith-eval (third expr))))))

