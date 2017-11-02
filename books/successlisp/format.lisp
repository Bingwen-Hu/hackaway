;; return string to a variable
(defparameter *s*
  (make-array 0
	      :element-type 'character
	      :adjustable t
	      :fill-pointer 0))
(format *s* "Hello~%")
(prin1 *s*)
(format *s* "Goodbye")
(print *s*)
(setf (fill-pointer *s*) 0)
(format *s* "A new beginning")
(print *s*)


;; interesting
(format nil "~R" 7)
(format nil "~R" 376)
(format nil "~:R" 7)
(format nil "~@R" 1999)

;; plurals
(format nil "~D time~:P, ~D fl~:@P" 1 1)
(format nil "~D time~:P, ~D fl~:@P" 3 5)


;; iteration
(format t "~&Name~20TExtension~{~&~A~20T~A~}"
	'("Joe" 3215 "Mary" 3246 "Fred" 3222 "Dave" 3232 "Joseph" 3212))

;; conditions
(format t "~[Lisp 1.5~; MACLISP~; PSL~; Common Lisp~]" 2)

(format t "My computer ~:[doesn't~;does~] like Lisp." t)
(format t "My computer ~:[doesn't~;does~] like Lisp." nil)

;; floobydust
(format t "~{~&~VD~}" '(5 37 10 253 15 9847 10 559 5 12)))