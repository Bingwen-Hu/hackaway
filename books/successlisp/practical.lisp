;;;; declaration
;; SPECIAL OPTIMIZE DYNAMIC-EXTENT TYPE FTYPE IGNORE IGNORABLE INLINE 
;; NOTINLINE


;; DEFVAR vs DEFPARAMETER
;; DEFVAR declare values that only need to evaluate once and 
;; DEFPARAMETER declare value every time it evaluate.


;; DefConstant define Constant
;; an available convention: +MY-CONSTANT+


;; Effciency: spotting, testing it
(defun sum-list-bad-1 (list)
  (let ((result 0))
    (dotimes (i (length list))
      (incf result (elt list i)))
    result))

;; recursive tail-call optimization
(defun sum-list-bad-2 (list)
  (labels ((do-sum (rest-list sum)
	     (if (null rest-list)
		 sum
		 (do-sum (rest rest-list) (+ sum (first rest-list))))))
    (do-sum list 0)))

;; fail tail-call optimization
(defun sum-list-bad-3 (list)
  (declare (optimize (debug 3)))
  (labels ((do-sum (rest-list sum)
	     (if (null rest-list)
		 sum
		 (do-sum (rest rest-list) (+ sum (first rest-list))))))
    (do-sum list 0)))


(defun sum-list-good (list)
  (let ((sum 0))
    (do ((list list (rest list)))
	((endp list) sum)
      (incf sum (first list)))))

;; test every function
(let ((list (make-list 10000 :initial-element 1)))
  (time (sum-list-bad-3 list)))