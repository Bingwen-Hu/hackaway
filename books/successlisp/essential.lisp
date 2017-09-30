;; atoms
(atom 123)
(numberp 123)
(atom :foo)
(numberp :foo)


;; Lisp is case-insensitive
(values 1 2 3 :hi "Mory")
(VALUES 1 2 3 :HI "mory")


;; assignment
(setq my-name "Mory")
(setq my-partner "Ann")
(setq book "successlisp" Author "David B. Lamkins" reader "Mory")

;; note that: the first argument of SETQ is a symbol, will not be evaluated.
;; if evaluate the first argument is what expected, using SET


;; local scope variable
(let ((a 3)
      (b 4)
      (c 5))
  (* (+ a b) c))

;; Note That: LET performs parallel binding, LET* performs ordinal binding.

;; Condition
;; Note: COND compare in order and will not falling down!
(let ((a 32))
  (cond ((eql a 13) "An unlucky number")
	((eql a 99) "A lucky number")
	(t "Nothing special about this number")))


;; QUOTE suppress evaluation rules.
(setq a 97)
(setq b 42)
(setq a b)				;=> 42
(setq a (quote b))			;=> B
(eql a b) 				;=> NIL


;; CONS construct a list
(cons 1 nil)
(cons 2 (cons 1 nil))
(cons 3 (cons 2 (cons 1 nil)))
(eq () NIL)				;=> T
(eql () NIL)				;=> T

;; LIST a function to construct a list
(list 1 2 3)
(list 1 :Mory "Ann" 'Jenny)


;; first and rest
(setq my-list '(1 2 3 4 5))
(first my-list)
(second my-list)


;; Naming and Identity
;; A symbol is always unique
(eq 'Mory 'mory)			;=> T
(setq zzz 'sleeper)
(eq zzz 'sleeper)

;; binding and shadow rules in CL is intutive

;; DEFUN
(defun secret-number (the-number)
  (let ((the-secret 42))
    (cond ((= the-number the-secret) 'You-win!)
	  ((< the-number the-secret) 'Too-small)
	  ((> the-number the-secret) 'Too-large))))

;; anonymous function
;; Note: lambda cannot be evaluated, only used as a function
((lambda (a b c x)
   (+ (* a x) (- b c)))
 3 4 5 7)


;; Macros return a form, not values
(defmacro setq-literal (place literal)
  `(setq ,place ',literal))
(setq-literal a b)			;=> B
(defmacro reverse-cons (rest first)
  `(cons ,first ,rest))
(reverse-cons nil A)			;=> (B)

(macroexpand '(setq-literal a b))
(macroexpand '(reverse-cons nil a))

;; Note: macroexpand is a funtion, so the form should be quoted


;; multivalues with VALUES

(values)
(values :this :that)
;; Note: VALUES is a function, its arguments are evaluated first

;; multiple-value-bind do the deconstruct
(multiple-value-bind (a b c) (values 2 3 5)
  (+ a b c))				;=> 10
(multiple-value-bind (a b c) (values 2 3 5 'x 'y) ; ignore excess values
  (+ a b c))				;=> 10

(multiple-value-bind (a b c) (values 2 3)
  (list a b c))				;=> (2 3 NIL)


(defun bookself (book author)
  (values (list :book book) (list :author author)))

(let ((author "Mory")
      (book "surpasser"))
  (bookself book author))
      

;; Data types
;; float
;; rationals
;; bignums
;; fixnums

;; Array organize data into tables
(setq al (make-array '(3 4)))
(setf (aref al 0 0) (list 'element 0 0))
(aref al 0 0)
(setf (aref al 0 1) pi)

;; Note: SETF and SETQ is similar, while SETF assigns a value to a place, 
;;       SETQ assigns a value to a symbol

(setq vec2 (vector 34 56 30))		;Vector form
(setq vec (make-array '(3)))
(equalp vec (vector 0 0 0))		;=> T
(equal vec (vector 0 0 0))		;=> NIL


(aref vec2 1)
(elt vec2 1)
;; ELT is special to sequence


;; STRING are vectors that contain only characters
;; so they are mutable
(setq s1 "Hello, there.")
(setf (elt s1 0) #\G)
(string 'symbol)
(string 'a-42)

;; Note: symbol can only start with alphabet

;; Symbol is complicate object
(setf (get 'Mory 'NickName) 'Demon)
(setf (get 'Mory 'SKill) 'Coding)
(setf (get 'Mory 'Love) 'Ann)
(symbol-plist 'Mory)
(get 'Mory 'Love)

;; Note: symbol properties are less often used in modern Lisp programs.

;; Structures in Lisp
;; defstruct will create a series of getter of the structure
(defstruct person name age gender)
(setq mory (make-person
	    :name 'Mory
	    :age 24
	    :gender 'Male))
(person-age mory)


;; Type
(type-of 123)
(type-of 123456678)
(type-of "Jenny Hello")
(type-of 'Mory)
(type-of (list 1 2 3))			;=> Cons



;; HASH TABLE
(setq dict (make-hash-table))
(gethash 'quux dict)
(setf (gethash 'name dict) 'Mory)
(gethash 'name dict)
(remhash 'name dict)

