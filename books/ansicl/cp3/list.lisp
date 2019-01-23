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


; access
(nth 0 '(1 2 3))
(nthcdr 0 '(1 2 3))
(nthcdr 1 '(1 2 3))


; mapping
; mapcar => map car
(mapcar #'(lambda (x y) (list x y)) 
        '(1 2 3)
        '(a b c d))
; maplist => map cdr
(maplist #'(lambda (x) x) '(a b c d))

; tree
; Conses can also be considered as binary trees, with the car representing the right subtree and the cdr the left

(defun our-copy-tree (tr)
  (if (atom tr)
      tr
      (cons (our-copy-tree (car tr))
            (our-copy-tree (cdr tr)))))


; replace element in a tree
; this will not work
(substitute 'y 'x '(and (integerp x) (zerop (mod x 2))))
(subst 'y 'x '(and (integerp x) (zerop (mod x 2))))

(defun our-subst (new old tree)
  (if (eql tree old)
      new
      (if (atom tree)
          tree
          (cons (our-subst new old (car tree))
                (our-subst new old (cdr tree))))))



; list as set
(member 'b '(a b c))
(defun mory-member (item lst)
  (if (null lst)
      nil
      (if (eql item (car lst))
          lst
          (mory-member item (cdr lst)))))

; introduce keyword parameter
(member '(a) '((a) (z)) :test #'equal)

(member-if #'oddp '(1 2 3 4))
(mapcar #'oddp '(1 2 3 4))

(defun our-member-if (fn lst)
  (and (consp lst)
       (if (funcall fn (car lst))
           lst
           (our-member-if fn (cdr lst)))))

(adjoin 'b '(a b c))
(adjoin 'z '(a b c))


(union '(a b c) '(c b s))
(intersection '(a b c) '(b b c))
(set-difference '(a b c d e) '(b e))

;;; 3.11 sequences
(length '(a b c))

(setf lst '(a b c d))
(setf sublst (subseq lst 1 3))
(setf sublst 'Mory)
(format t "~a~%" sublst)

; palindrome
(defun mirror? (s)
  (let ((len (length s)))
    (and (evenp len)
         (let ((mid (/ len 2)))
           (equal (subseq s 0 mid)
                  (reverse (subseq s mid)))))))
(defun )
