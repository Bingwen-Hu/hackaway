;;; reverse
(defun good-reverse (lst)
  (labels ((rev (lst acc)
             (if (null lst)
                 acc
                 (rev (cdr lst) (cons (car lst) acc)))))
    (rev lst nil)))

; multivalues
(truncate 3.14)
(multiple-value-bind (int frac) (truncate 3.14)
  (list int frac))
