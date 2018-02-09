;; basic file io
(defparameter path "E:/Mory/github/hackaway/books/PractialCL/filesIO.lisp")
(let ((in (open path)))
  (format t "~a~%" (read-line in))
  (close in))

;; read the whole file
(let ((in (open path :if-does-not-exist nil)))
  (when in 
    (loop for line = (read-line in nil)
	 while line do (format t "~a~%" line))
    (close in)))

;; read S-expression
(defparameter path "E:/Mory/github/hackaway/books/PractialCL/function.lisp")
(with-open-file (in path)
  (dotimes (i 7)
    (eval (read in))))


;; print -> s-expression
;; prin1 -> just he s-expression
;; pprint -> beautiful print


;; big topic on pathname