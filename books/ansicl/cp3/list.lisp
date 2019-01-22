;;; 3.1 conses

; cons to nil
(setf x (cons 'a nil))

(car x)
(cdr x)

; list construction
(setf y (list 'a 'b 'c))
(cdr y)

; nest list
(setf z (list 'a (list 'b 'c) 'd))
(car (cdr z))


; listp
(defun mory-listp (x)
  (or (null x) (consp x)))

; everything is not a atom is a cons
(defun mory-atom (x)
  (not (consp x)))


; eql test whether two is the same object
; equal test whether two object contain same value
(eql 'a 'a)
(equal '(lst 1) '(lst 1))
(eql '(lst 1) '(lst 1))

(defun mory-equal (x y)
  (or (eql x y)
      (and (consp x)
           (consp y)
           (mory-equal (car x) (car y))
           (mory-equal (cdr x) (cdr y)))))

; copy list
(let* ((x '(a b c))
       (y (copy-list x)))
  (format t "eql test: ~a~%" (eql x y))
  (format t "equal test: ~a~%" (equal x y)))

(defun mory-copy-list (lst)
  (if (atom lst)
      lst
      (cons (car lst) (mory-copy-list (cdr lst)))))

(defun mory-copy-list (lst)
  (labels ((helper (lst ret)
             (if (null lst)
                 ret
                 (helper (cdr lst) (cons (car lst) ret)))))
    (nreverse (helper lst nil))))

(append '(a c) '(c d) '(e))

(defun mory-append (&rest lst)
  (if (null lst)
      nil
      (let ((x nil))
        (dolist (lst-* lst)
          (dolist (i lst-*)
            (setf x (cons i x))
            (format t "x: ~a~%" x)))
        (nreverse x))))

(defun mory-append ())
