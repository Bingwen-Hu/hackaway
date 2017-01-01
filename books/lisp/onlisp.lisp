;; what's macro?
(defmacro our-expander (name) `(get ,name 'expander))

(defmacro our-defmacro (name parms &body body)
  (let ((g (gensym)))
    `(progn
       (setf (our-expander ',name)
	     #'(lambda (,g)
		 (block ,name
		   (destructuring-bind ,parms (cdr ,g) ; where is car g?
		     ,@body))))
       ',name)))

(defun our-macroexpand-1 (expr)
  (if (and (consp expr) (our-expander (car expr)))
      (funcall (our-expander (car expr)) expr)
      expr))


;; do macro
(defmacro our-do (bindforms (test &rest result) &body body)
  (let ((label (gensym)))
    `(prog ,(make-initforms bindforms)
	,label
	(if ,test
	    (return (progn ,@result)))
	,@body
	(psetq ,@(make-stepforms bindforms))
	(go ,label))))

(defun make-initforms (bindforms)
  (mapcar #'(lambda (b)
	      (if (consp b)
		  (list (car b) (cadr b))
		  (list b nil)))
	  bindforms))
			
(defun make-stepforms (bindforms)
  (mapcan #'(lambda (b)
	      (if (and (consp b) (third b))
		  (list (car b) (third b))
		  nil))
	  bindforms))
