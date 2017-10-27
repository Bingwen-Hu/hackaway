;; using disassemble to see whether declarations have any effect on your compiler
(defun add1 (n) (1+ n))
(disassemble 'add1)

;; make some declare
(defun int-add1 (n)
  (declare (fixnum n)
	   (optimize (speed 3) (safety 0) (debug 0)))
  (the fixnum (1+ n)))

;; STEP as the useful function