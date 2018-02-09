;;; basic parts
(cons 1 2)
(cons 1 nil)
(cons 2 (cons 1 nil))
(cons 3 (cons 2 (cons 1 nil)))
(list 1)
(list 1 2)
(list 1 2 3)


;;; functional programming style
(append (list 1 2) (list 3 5))

;;; destructive operations append->nconc
(nconc (list 1 2 3) (list 2 4))
; similarly
; remove -> delete
; substitute -> nsubstitute
; reverse -> nreverse

; recycling idiom
(defun upto (max)
  (let ((result nil))
    (dotimes (i max)
      (push i result))
    (nreverse result)))

(defparameter *list*)
(setf *list* (reverse (list 1 2 3 4 5)))


;;; list manipulation functions
(defvar my-list '(2 3 5 6 7)
  "global lisp")
(cadr my-list)
; caar, caadr, ....

(last my-list)
(butlast my-list)
(ldiff my-list 2) ; ?
(make-list 4 :initial-element 8)

; list* is a cross between list and append
(list* 1 2 3 4 '(mory ann jenny))
(list 1 2 3 4 'Mory 'Ann 'Jenny)
(append '(1 2 3 4) '(Mory Ann Jenny))

;revappend: reverse first argument and append to the second
(revappend '(a b c) '(h i j))
;revappend -> nreconc
(consp nil)
(consp my-list)
(listp nil)
(listp my-list)

;;; mapping
(mapcar #'(lambda (x) (* 2 x)) '(1 2 3))
(mapcar #'+ (list 1 2 3) (list 20 3 20))
(maplist #'(lambda (l1 l2) (append l1 l2)) '(1 2 3) '(4 5 6))
; mapcan mapcon

