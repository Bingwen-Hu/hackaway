;;; symbol in Common Lisp is case-insensitive
;;; special symbol in other language is useable in Lisp
;;; such as $ -> and so on.

(print 'John)
(print '(John Q Public))
(print '(+ 2 2))
(print (+ 2 2))

(append '(Pat Kim) (list '(John Q Public) 'Sandy))
(length '(Pat Kim (list '(Haha J Mory) 'Ann)))
