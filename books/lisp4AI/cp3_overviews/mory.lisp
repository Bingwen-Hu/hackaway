(defun maximum (lst)
  (labels ((helper (lst max) 
             (if (null lst)
                 max
                 (let ((temp (first lst)))
                   (if (< temp max)
                       (helper (rest lst) max)
                       (helper (rest lst) temp))))))
    (helper lst -1)))

(defun maximum-cond (lst &optional (max -1))
  (cond ((null lst) max)
        ((< (first lst) max)
         (maximum-cond (rest lst) max))
        (t (maximum-cond (rest lst) (first lst)))))

(defparameter lst '(1 2 3 84 3 4 2 42))
(maximum-cond lst)
