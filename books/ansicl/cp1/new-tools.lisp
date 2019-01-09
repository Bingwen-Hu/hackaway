;;; 1.1 New Tools

; you can't do this in C
(defun addn (n)
  #'(lambda (x)
      (+ x n)))

; annoy lisp2
(funcall (addn 3) 42)

;;; 1.2 New Techniques


;;; 1.3 A New Approach

; plan-and-implement model sounds good in theory
; but does not work very well

; Instead of hoping that people won't make mistakes, we 
; try to make the cost of mistakes very low

; Planning is a necessary evil

