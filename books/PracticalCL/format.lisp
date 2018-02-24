(loop for cons on (list 1 2 3 4 5)
   do (format t "~a" (car cons))
   when (cdr cons) do (format t ", "))

;equals to
(format t "~{~a~^, ~}" (list 1 2 3 4 5))

;format a floating-point number
(format t "~$" pi)
(format t "~5$" pi)
(format t "~v$" 4 pi)
(format t "~#$" pi)
(format t "~,5f", pi)
(format t "~d" 1000000)
(format t "~:d" 1000000)
(format t "~@d" 1000000)
(format t "~:@d" 1000000)

;;; basic formatting
(format nil "The value is: ~a" 10)
(format nil "The value is: ~a" "foo")
(format nil "The value is: ~a" (list 1 2 3))

;;; char
(format t "~c" #\a)
(format t "~:c" #\a)
(format t "~@c" #\a)

;;; integer
(format nil "~12d" 1000000)
(format nil "~12,'0d" 1000000)
(format nil "~12,:d" 1000000)
(format nil "~4,'0d-~2,'0d-~2,'0d" 2005 5 10)

(format nil "~x" 1000000)
(format nil "~o" 1000000)
(format nil "~b" 1000000)

(format nil "~f" pi)
(format nil "~,4f" pi)
(format nil "~e" pi)
(format nil "~,4e" pi)
(format nil "~2,4$" pi)

(format nil "~r" 1234)

;;; conditional formatting
(format nil "~[cero~;uno~;dos~]" 0)
(format nil "~[cero~;uno~;dos~]" 1)
(format nil "~[cero~;uno~;dos~]" 2)

;until 
(format nil "~[good~;soso~:;bad...~]" 2)
(format nil "~[good~;soso~:;bad...~]" 20)

;;; an example
(defparameter *list-etc*
  "~#[NONE~;~a~;~a and ~a~:;~a, ~a~]~#[~; and ~a~:;, ~a, etc~].")

(format nil *list-etc*)
(format nil *list-etc* 'a)
(format nil *list-etc* 'a 'b)
(format nil *list-etc* 'a 'b 'c)
(format nil *list-etc* 'a 'b 'c 'd)
(format nil *list-etc* 'a 'b 'c 'd 'e)


;;; iteration
(format nil "~{~a, ~}" '(1 2 3 4))
(format nil "~{~a~^, ~}" '(1 2 3 4))
(format nil "~@{~a~^, ~}" 1 2 3)


