;;; AI in lisp

;; set in common lisp

;; probability theory
(defun uniform (a b)
  (let ((domain (- b a)))
    (lambda (x)
      (cond ((> x b) 1)
            ((< x a) 0)
            (t (/ (- x a)
                  domain))))))
(funcall (uniform 1 4) 4)
;;
