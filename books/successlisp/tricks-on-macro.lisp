;; keyword arguments
(defun keyword-sample (a b c &key (d 12) (e 32) (f 43))
  (list a b c d e f ))

(keyword-sample 1 2 3)
(keyword-sample 1 2 3 :d 4)
(keyword-sample 1 2 3 4 5 6)		;error
(keyword-sample 1 2 3 :d 4 :f 9 :e 23)

