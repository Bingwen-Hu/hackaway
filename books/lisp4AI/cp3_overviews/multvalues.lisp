(defun show-both (x)
  (multiple-value-bind (int rem)
      (round x)
    (format t "~f = ~d + ~f" x int rem)))

(show-both 5.1)
