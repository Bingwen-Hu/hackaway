;; basic file io
(defparameter path "E:/Mory/github/hackaway/books/PractialCL/filesIO.lisp")
(let ((in (open path)))
  (format t "~a~%" (read-line in))
  (close in))