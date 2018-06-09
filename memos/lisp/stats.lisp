(defun sum (lst)
  (if (null lst)
      0
      (+ (car lst) (sum (cdr lst)))))

(defun len (lst)
  (if (null lst)
      0
      (+ 1 (len (cdr lst)))))

(defun mean (lst)
  (/ (sum lst) (len lst)))


(defun power (x n)
  (labels ((helper (x n result)
	     (if (= n 0)
		 result
		 (helper x (1- n) (* x result)))))
    (helper x n 1)))

(defun variance (lst) 
  (let* ((m (mean lst))
	(res (loop for i in lst 
		collect (power (- i m) 2))))
    (sum res)))

;; std-deviation could not be implement because 
;; I don't know how to implement deviation