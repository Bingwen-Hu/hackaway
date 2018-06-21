(with-open-file (stream "test" :direction :output)
  (print '(hello there) stream)
  (princ "Good bye stream" stream))
