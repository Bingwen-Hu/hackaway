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

;;; example
(defun primep (number)
  (when (> number 1)
    (loop for fac from 2 to (isqrt number) 
       never (zerop (mod number fac)))))

(defun next-prime (number)
  (loop for n from number when (primep n) return n))


(defmacro do-primes (var-and-range &rest body)
  (let ((var (first var-and-range))
        (start (second var-and-range))
        (end (third var-and-range)))
    `(do ((,var (next-prime ,start) (next-prime (1+ ,var))))
         ((> ,var ,end))
       ,@body)))

;;; better version
(defmacro do-primes2 ((var start end) &body body)
  `(do ((,var (next-prime ,start) (next-prime (1+ ,var))))
       ((> ,var ,end))
     ,@body))

;;; Introduce of Principle of Least Astonishment

;;; a better and better version fix to parameter leaky
(defmacro do-primes3 ((var start end) &body body)
  `(do ((,var (next-prime ,start) (next-prime (1+ ,var)))
        (ending-value ,end))
       ((> ,var ending-value))
     ,@body))


;;; a 3x better version to fix namely shadow
(defmacro do-primes4 ((var start end) &body body)
  (let ((ending-value-name (gensym)))
    `(do ((,var (next-prime ,start) (next-prime (1+ ,var)))
          (,ending-value-name ,end))
         ((> ,var ,ending-value-name))
       ,@body)))
