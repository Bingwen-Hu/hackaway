(sqrt -1)

(defun quadratic-roots (a b c)
  "Return the roots of a quadratic equation aX^2 + bX + c = 0,
Returns only one value if the roots are coincident."
  (let ((discriminant (- (* b b) (* 4 a c))))
    (cond ((zerop discriminant)
           ;; coincident roots -- return one value
           (/ (+ (- b) (sqrt discriminant)) (* 2 a)))
          (t 
           ;; two distinct roots
           (values (/ (+ (- b) (sqrt discriminant)) (* 2 a))
                   (/ (- (- b) (sqrt discriminant)) (* 2 a)))))))



;;; Optional parameters
(defun silly-list (p1 p2 &optional p3 p4)
  (list p1 p2 p3 p4))

;;; infinite parameters
(defun silly-list-2 (p1 p2 &rest p3)
  (list p1 p2 p3))


;;; combine two above
(defun silly-list-3 (p1 p2 &optional p3 p4 &rest p5)
  (list p1 p2 p3 p4 p5))


;;; define global parameter
(defvar *mory* 'Good) 
(defvar *mory* 'bad)
;;; Note that: Defvar only affect the nonbound symbol

(defparameter *Ann* 'beautiful "I think Ann is very beautiful")
(documentation '*Ann* 'variable)
;;; DEFPARAMETER is preferred for its documention form


;;; constants in lisp
(defconstant Ann 'surpasser "Ann is the leader of lighting surpasser")



;;; recursive functions two conditions
;;; 1. One case must not make a recursive call.
;;; 2. Other cases must reduce the amount of work to be done in a recursive call.

(defun factorial (n) 
  (cond ((zerop n) 1)
        (t (* n (factorial (1- n))))))

(defun mory-length (lst)
  (cond ((null lst) 0)
        (t (1+ (mory-length (rest lst)))))) 

;;; trace a recursive function
(trace mory-length)


;;; tail recursion
(defun factorial-tr (n)
  (factorial-tr-helper n 1))
(defun factorial-tr-helper (n product)
  (cond ((zerop n) product)
        (t (factorial-tr-helper (- n 1) (* product n)))))



;;; tricky code
(defun foo () 1)                        ;a function return 1

(defun baz ()
  (flet ((foo () 2)                     ;FLET bind a function
         (bar () (foo)))
    (values (foo) (bar))))

(defun raz ()
  (labels ((foo () 2)
           (bar () (foo)))
    (values (foo) (bar))))

;; Note: FLET and LABLES is just like LET and LET*
;; FLET is paralleled and labels is orderial



;;; Reading, writing, and arithmetic

(defun simple-adding-machine-1 ()
  (let ((sum 0)
        next)
    (loop 
       (setq next (read))
       (cond ((numberp next)
              (incf sum next))
             ((eq '= next)
              (print sum)
              (return))
             (t 
              (format t "~&~A ignored!~%" next))))
    (values)))


; what happen if I want to redirect the input and output stream?
(with-open-file (in-stream *infile.bat* :direction :input)
  (with-open-file (out-stream *outfile.bat* :direction :output)
    (let ((*standard-input* in-stream)
          (*standard-output* out-stream))
      (declare (special *standard-input* *standard-output*))
      (simple-adding-machine-1))))
;;; This is useful when the SIMPLE-ADDING-Machine is not written by yourself

;;; Or we can design at the very beginning, using default parameters
(defun simple-adding-machine-2 (&optional (in-stream *standard-input*)
                                  (out-stream *standard-output*))
  (let ((sum 0)
        next)
    (loop 
       (setq next (read in-stream))
       (cond ((numberp next)
              (incf sum next))
             ((eq '= next)
              (print sum out-stream)
              (return))
             (t (format out-stream "~&~A ignored!~%" next))))
    (values)))

; and call as this
(with-open-file (in-stream "infile.dat" :direction :input)
  (with-open-file (out-stream "outfile.dat" :direction :output)
    (simple-adding-machine-2 in-stream out-stream)))
