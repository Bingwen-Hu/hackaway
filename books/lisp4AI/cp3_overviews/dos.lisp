(defparameter x '(1 2 3 4 5))

(defun length-1 (lst)
  (let ((len 0))
    (dolist (i lst len)
      (incf len))))

(defun length-2 (lst)
  (do ((len 0 (+ len 1))
       (l lst (rest l)))
      ((null l) len)))


(defun length-3 (lst)
  (loop for element in lst
       count element))

(defun length-4 (lst)
  (loop for element in lst
       summing 1))

(defun length-5 (lst)
  (loop with len = 0
       until (null lst)
       for element = (pop lst)
       do (incf len)
       finally (return len)))

