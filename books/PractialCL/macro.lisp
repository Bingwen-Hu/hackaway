;;; macro

;;; unless conflict with exist function/macros
(defmacro unless1 (condition &rest body)
    `(if (not ,condition)
         (progn ,@body)))

(defmacro my-when (condition &rest body)
  `(if ,condition
       (progn ,@body)))

;;; dotimes dolist

(do ((n 0 (1+ n))
     (cur 0 next)
     (next 1 (+ cur next)))
    ((= 10 n) cur))

(defparameter sum 0)
(dotimes (i 10)
  (incf sum i))


;;; loop
(loop for i from 1 to 10 collecting i)
(loop for i from 1 to 10 summing i)


;;; define mory macros


