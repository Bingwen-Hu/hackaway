;; a tool to trace the macro
;; PPMX stands for "Pretty Print Macro eXpansion"
;; thanks for David <Common Lisp> book

(defmacro ppmx (form)
  "Pretty prints the macro expansion of FORM."
  `(let* ((exp1 (macroexpand-1 ',form))
	  (exp (macroexpand exp1))
	  (*print-circle* nil))
     (cond ((equal exp exp1)
	    (format t "~&Macro expansion:")
	    (pprint exp))
	   (t (format t "~&First step of expansion:")
	      (pprint exp1)
	      (format t "~%~%Final expansion:")
	      (pprint exp)))
     (format t "~%~%")
     (values)))

;; a trick to read macro
(defmacro two-from-one (func object)
  `(,func ',object ',object))

;; ',object can be consider as  (quote ,object)
;; first, eval ,object get the value pass in
;; then, quote it as a symbol
;; so ,SYMBOL means the value return by (eval symbol)
;; ',SYMBOL just quote the value as a symbol!

;; splicing the ,@ sugar

;; `(,name lives at ,address now)
;; (FRED LIVES AT (16 MAPLE DRIVE) NOW)
;; `(,name lives at ,@address now)
;; (FRED LIVES AT 16 MAPLE DRIVE NOW)
