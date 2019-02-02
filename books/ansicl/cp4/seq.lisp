;;; for vector, which allow random access
(defun mirror? (s)
  (let ((len (length s)))
    (and (evenp len)
         (do ((forward 0 (+ forward 1))
              (back (- len 1) (- back 1)))
             ((or (> forward back)
                  (not (eql (elt s forward)
                            (elt s back))))
              (> forward back))))))


;;; keyword parameters
(position #\a "fantasia")
(position #\a "fantasia" :start 3 :end 5)
(position #\a "fantasia" :from-end t)
(position 'a '((c d) (a b)) :key #'car)
(position '(a b) '((a b) (c d)))
(position '(a b) '((a b) ()))

(defun second-word (str)
  (let* ((start (1+ (position #\  str)))
        (end (position #\  str :start start)))
    (subseq str start end)))

(find #\a "cat")

(find-if #'characterp "ham")

