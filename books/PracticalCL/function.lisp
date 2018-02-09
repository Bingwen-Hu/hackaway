;;; &optional parameter

(defun make-rectangle (width &optional (height width))
  (format t "make rectangle width ~A height ~A" width height))

;;; what about keyword?
(defun make-surpasser (&key kind level)
  (format nil "~A surpasser in level ~A" kind level))
;;; Test Note
;;; only keyword parameter accept!


(function make-surpasser)               ;=> #'make-surpasser
(make-surpasser :kind 'Flame :level 4) 
(funcall #'make-surpasser :kind 'Flame :level 4)
(apply #'make-surpasser '(:kind 'Flame :level 4))


;;; a taste for loop
(loop for i from 0 to 10 by 2 do
     (format t "~a ~%" i))
