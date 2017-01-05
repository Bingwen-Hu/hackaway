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

;; two version of and
;; easy to read but low effection
(defmacro our-and (&rest args)
  (case (length args)
    (0 t)
    (1 (car args))
    (t `(if ,(car args)
	    (our-and ,@(cdr args))))))
       
;; high performance not very intution
(defmacro our-andb (&rest args)
  (if (null args)
      t
      (labels ((our-expander (rest)
		 (if (cdr rest)
		     `(if ,(car rest)
			  ,(our-expander (cdr rest)))
		     (car rest))))
	(our-expander args))))


;; let's insert something new: Symbol-macro
(+ (progn (print "Howdy") 1) 2) 	; => "Howdy" 3
(symbol-macrolet ((hi (progn (print "Howdy") 1))))
;; symbol-macro 'hi' is replace every where by its value


;; another with-gensym

(defmacro with-gensyms (syms &body body)
  `(let ,(mapcar #'(lambda (s)
		     `(,s (gensym)))
		 syms)
     ,@body))
