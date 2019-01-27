; Write a version of union that preserves the order of the elements in the original lists


; Define a function that takes a list and returns a list indicating the number of times each ( eql) element appears, sorted from most common element to least common
(defun occurrences (lst)
  (labels ((occur (lst ret)
             (if (null lst)
                 ret
                 (let ((next (car lst)))
                   (if (assoc next ret)
                       (progn (incf (cdr (assoc next ret)))
                              (occur (cdr lst) ret))
                       (progn (push `(,next . 1) ret)
                              (occur (cdr lst) ret)))))))
    (let ((ret (occur lst nil)))
      (sort ret #'> :key #'cdr))))

