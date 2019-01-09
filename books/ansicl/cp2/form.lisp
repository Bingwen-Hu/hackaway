; 2.1 form

; evaluate
(+ 3 5)
; and not
'(+ 3 5)

; symbol
'Artichoke

; lists
'(my 3 "sons")

'(the list (abd) has 3 elements)

(list 'my (+ 2 1) "sons")

; Lisp programs are expressed as lists
; So we need quote.

(list '(+ 2 1) (+ 2 1))

; two empty list representation
(listp ())
(listp nil)

; 2.4 list operations

; cons
(cons 'a '(b c d))
(cons 'a (cons 'b nil))
(list 'a 'b)
(car '(a b c))
(cdr '(a b c))

(third '(a b c d))

; 2.5 Truth
(listp '(a b c))
(listp 27)
(null nil)
(not nil)


;; mory note
;; If you have something to worry, you have no happiness
