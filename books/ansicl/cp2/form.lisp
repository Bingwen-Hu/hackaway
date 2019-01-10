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
(if (listp '(a b c))
    (+ 1 2)
    (+ 4 6))

(if (listp 42)
    (+ 1 2)
    (+ 4 6))

; if is a special operator. It could not possibly be implemented as a function, because the arguments in a function call are always evaluated and the whole point of if is that only one of the last two arguments is evaluated

(if (listp 42)
    (+ 2 4))


; macro
(and t (+ 1 2))
(or '() 8 4 nil)


;;; 2.6
; (defun function-name (list of parameters) (list of body)*)
(defun our-third (x)
  (car (cdr (cdr x))))

;;; 2.7 recursion

(defun our-member (obj lst)
  "test whether a obj in a lst"
  (if (null lst)
      nil
      (if (eql (car lst) obj)
          lst 
          (our-member obj (cdr lst)))))

; Think recursion as a rule not a machine

; 2.9 input and output
; small recipe
; ~a as a placeholder, valid for string, numeric, symbol
(format t "~a plus ~a equals ~a." 2 3 (+ 2 3))
; ~% as a newline
(format t "Love is~%needed by everyone.")


; read in
(let ((name (read)))
  (format t "my name is ~a" name))

(defun askem (string)
  (format t "~A" string)
  (read))

;;; 2.10 variables

; local variable
; (let ((var val)*) (body)*)
(let () ())
(let ((x 1)
      (y 2))
  (+ x y))

(defun ask-number () 
  (format t "Please enter a number. ")
  (let ((val (read)))
    (if (numberp val)
        val 
        (ask-number))))

; global parameters
(defparameter *glob* 42)
(defconstant limit (+ *glob* 1))

; check if the name is bounded
(boundp 'limit)

;;; 2.11 assignment

; general assignment
; create global variable implicitly
(setf *mory* 98)
(let ((n 10))
  (setf n 2)
  n)

; assign a position
(let ((x '(A B C)))
  (setf (car x) 'Ann)
  x)

(setf a 'a
      b 'b
      c 'c)


;;; 2.12 functional programming
(setf lst '(c a r a t))
(remove 'a lst)

(defun mory-remove (obj lst)
  "recurse process"
  (if (null lst)
      nil
      (if (eql (car lst) obj)
          (mory-remove obj (cdr lst))
          (cons (car lst) (mory-remove obj (cdr lst))))))

(defun mory-remove-iterate (obj lst)
  (labels ((remove-iter (obj lst ret)
             (cond ((null lst) ret)
                   ((eql (car lst) obj) 
                    (remove-iter obj (cdr lst) ret))
                   (t 
                    (remove-iter obj (cdr lst) (cons (car lst) ret))))))
    (nreverse (remove-iter obj lst))))

;;; 2.13 iteration
; (do ((variable initial update)) (end-condition return-value) body*)
(defun show-squares (start end)
  (do ((i start (+ i 1)))
      ((> i end) 'done)
    (format t "~a ~a~%" i (* i i))))

(defun mory-length (lst)
  (let ((len 0))
    (dolist (- lst)
      (setf len (1+ len)))
    len))

(defun mory-length-recursive (lst)
  (if (null lst)
      0
      (+ 1 (mory-length-recursive (cdr lst)))))

(defun mory-length-iterative (lst)
  (labels ((length-iter (lst len)
             (if (null lst)
                 len
                 (length-iter (cdr lst) (1+ len)))))
    (length-iter lst 0)))


;;; 2.14 functions as objects
