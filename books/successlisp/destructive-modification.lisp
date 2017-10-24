;; closure
(let ((e 1))
  (defun closure-1 () e)
  (setq e 7)
  (defun closure-2 () e))

(let ((counter 0))
  (defun counter-next ()
    (incf counter))
  (defun counter-reset ()
    (setq counter 0)))