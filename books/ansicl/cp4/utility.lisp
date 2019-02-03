;;; intution to recognize the lack of operator

; operations on lists
(proclaim '(inline last1 single append1 conc1 mklist))

(defun last1 (lst)
  (car (last lst)))

(defun single (lst)
  (and (consp lst) (not (cdr lst))))
;; compare to 
;; (= (length) 1)
;; single is fast for long list

; nconc => append 

(defun conc1 (lst obj)
  (nconc lst (list obj)))

(defun append1 (lst obj)
  (append lst (list obj)))

(defun mklist (obj)
  "make sure an object is list"
  (if (listp obj) obj (list obj)))

;;; call compile-file to this file generate a compile version
(defun longer (x y)
  (labels ((compare (x y)
             (and (consp x)
                  (or (null y)
                      (compare (cdr x) (cdr y))))))
    (if (and (listp x) (listp y))
        (compare x y)
        (> (length x) (length y)))))

;; push and nreverse is common pair
(defun filter (fn lst)
  (let ((acc nil))
    (dolist (x lst)
      (let ((val (funcall fn x)))
        (if val (push val acc))))
    (nreverse acc)))


;; group
(nthcdr 3 '(1 2 3 4 5 6 5))
(defun group (source n)
  (if (zerop n) (error "zero length"))
  (labels ((rec (source acc)
             (let ((rest (nthcdr n source)))
               (if (consp rest)
                   (rec rest (cons (subseq source 0 n) acc))
                   (nreverse (cons source acc))))))
    (if source (rec source nil) nil)))


;; flatten 
(defun flatten (x)
  (labels ((rec (x acc)
             (cond ((null x) acc)
                   ((atom x) (cons x acc))
                   (t (rec (car x) (rec (cdr x) acc))))))
    (rec x nil)))

(flatten '(a (b c) ((d e) f)))

(defun prune (test tree)
  (labels ((rec (tree acc)
             (cond ((null tree) (nreverse acc))
                   ((consp (car tree))
                    (rec (cdr tree)
                         (cons (rec (car tree) nil) acc)))
                   (t (rec (cdr tree)
                           (if (funcall test (car tree))
                               acc 
                               (cons (car tree) acc)))))))
    (rec tree nil)))

(prune #'evenp '(1 2 (3 (4 5) 6) 7 8 (9)))


;;; search
