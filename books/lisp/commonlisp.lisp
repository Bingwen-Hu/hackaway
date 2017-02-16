;;; Paradigms of AI programming 
;;; case studies in Common Lisp

;;; chapter1
;;; append one list to another list
(append '(Pat Kim) (list '(John Q Public) 'Sandy))

;;; a recursive function

(defparameter *title*
  '(Mr Mrs Miss Ms Sir Madam Dr Admiral Major General))

(defun first-name (name)
  "a function can skip the title and fetch first name"
  (if (member (first name) *title*)
      (first-name (rest name))
      (first name)))

;;; high order function

(defun mappend (fn the-list)
  "map fn on every item in the-list and append the result"
  (apply #'append (mapcar fn the-list)))

(defun self-and-double (x) (list x (+ x x)))
