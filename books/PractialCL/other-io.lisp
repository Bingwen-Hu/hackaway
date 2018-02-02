;;; string io
(let ((s (make-string-input-stream "1.23")))
  (unwind-protect (read s)
    (close s)))

;;; convenient macro for string reading
(with-input-from-string (s "1.23")
  (read s))

(with-output-to-string (out)
  (format out "hello, world ")
  (format out "~s" (list 1 2 3)))
