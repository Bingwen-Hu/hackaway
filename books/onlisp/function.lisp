;;; Functions
; in Lisp2 function and value and its own scope
(defun idouble (x) (* x 2))
(defvar idouble 2.0)
(symbol-function 'idouble)
(symbol-value 'idouble)


;; apply can take any number of arguments, and the function given first will be applied to the list made by consing the rest of the arguments onto the list given last
(apply #'+ 1 '(2))
(funcall #'+ 1 2)


;;; function as properties
; first approach
(defun behave (animal)
  (case animal 
    (dog (vag-tail)
         (bark))
    (rat (scurry)
         (squeak))
    (cat (rub-legs)
         (scratch-carpet))))
; second approach
; like dispatch in Python
(defun behave2 (animal)
  (funcall (get animal 'behavior)))

(setf (get 'dog 'behavior)
      #'(lambda ()
          (vag-tail)
          (bark)))


;;; classic example
(defun make-adder (n)
  #'(lambda (x) (+ x n)))

(setq add2 (make-adder 2)
      add10 (make-adder 10))

(funcall add2 5)
(funcall add10 3)


;;; more general
(defun make-adderb (n)
  #'(lambda (x &optional change)
      (if change
          (setq n x)
          (+ x n))))
(setq addx (make-adderb 1))
(funcall addx 3)
; change inner variable
(funcall addx 100 t)
(funcall addx 1)

;;; local functions
; labels' local function binding behavior just like let*

(defun count-instances (obj lsts)
  (labels ((instances-in (lst)
             (if (consp lst)
                 (+ (if (eq (car lst) obj) 1 0)
                    (instances-in (cdr lst)))
                 0)))
    (mapcar #'instances-in lsts)))

(count-instances 'a '((a b c a) (d a r p) (d a e) (a a)))


;; tail recursion optimizer
(proclaim '(optimize speed))

;;; fast common lisp
(defun triangle (n)
  (labels ((tri (c n)
             (declare (type fixnum n c))
             (if (zerop n)
                 c
                 (tri (the fixnum (+ n c))
                      (the fixnum (- n 1))))))
    (tri 0 n)))

(compile 'triangle)
(compiled-function-p #'triangle)
(compile nil '(lambda (x) (+ x 2)))


;; If you give both the name and function arguments, compile becomes a sort of
;; compiling defun                         
(progn (compile 'bar '(lambda (x) (* x 2)))
       (compiled-function-p #'bar))

;;; inline function
(defun 50th (lst) (nth 49 lst))
(proclaim '(inline 50th))
