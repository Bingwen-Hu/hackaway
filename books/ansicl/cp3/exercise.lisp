; Write a version of union that preserves the order of the elements in the original lists


; Define a function that takes a list and returns a list indicating the number of times each ( eql) element appears, sorted from most common element to least common
(defun occurrences (lst)
  (labels ((occur (lst ret)
             (if (null lst)
                 ret
                 (let ((next (car lst)))
                   (if (assoc next ret)
                       (progn (incf (cdr (assoc next ret)))
                              (occur (cdr lst) ret))
                       (progn (push `(,next . 1) ret)
                              (occur (cdr lst) ret)))))))
    (let ((ret (occur lst nil)))
      (sort ret #'> :key #'cdr))))

(occurrences '(a b a d a c d c a))

;;; e4
(member '(a) '((a) (b)))
(member '(a) '((a) (b)) :test #'equal)


;;; e5
(defun pos+ (lst)
  "iteration"
  (let ((index 0))
    (dolist (int lst lst)
      (setf (nth index lst) (+ index int))
      (incf index))))

(defun pos+ (lst)
  "recursion"
  (labels ((pos-add (to-add lst ret)
             (if (null lst)
                 ret
                 (pos-add (1+ to-add) (cdr lst) 
                          (cons (+ (car lst) to-add) ret)))))
    (nreverse (pos-add 0 lst nil))))

(defun pos+ (lst)
  "USING MAPCAR"
  (labels ((build-index (index lst ret)
             (if (null lst)
                 ret
                 (build-index (1+ index) (cdr lst) (cons index ret)))))
    (let ((index-list (nreverse (build-index 0 lst nil))))
      (mapcar #'+ lst index-list))))

; define your own cons, list, length, member
(defun mory-cons (a b)
  `(,a . ,b))

(defun mory-list (&rest lst)
  (defun list-with-ret (lst ret)
    (if (null lst)
        ret
        (list-with-ret (cdr lst) 
                       (mory-cons (car lst) ret))))
  (nreverse (list-with-ret lst nil)))

(defun mory-length (lst)
  (if (null lst)
      0
      (1+ (mory-length (cdr lst)))))

(defun mory-member (a lst)
  (if (null lst)
      nil
      (if (eql a (car lst))
          lst
          (mory-member a (cdr lst)))))

; defina a function that takes a list and prints it in dot notation
(defun showdots (lst)
  (if (null lst)
      (princ nil)
      (progn 
        (princ "(")
        (princ (car lst))
        (princ " . ")
        (showdots (cdr lst))
        (princ ")")
        nil)))
