;; keyword arguments
(defun keyword-sample (a b c &key (d 12) (e 32) (f 43))
  (list a b c d e f ))

(keyword-sample 1 2 3)
(keyword-sample 1 2 3 :d 4)
(keyword-sample 1 2 3 4 5 6)		;error
(keyword-sample 1 2 3 :d 4 :f 9 :e 23)

;; check whether is pass
(defun keyword-sample-3 (a &key (b nil b-p) (c 53 c-p))
  (list a b b-p c c-p))
(keyword-sample-3 1)
(keyword-sample-3 1 :b 74)
(keyword-sample-3 1 :b nil)

;; defaults value and supplied-p variable can use with &optional
(defun optional-sample-1 (a &optional (b nil b-p))
  (list a b b-p))

(optional-sample-1 1)
(optional-sample-1 1 nil)


;; &optional precede &key
(defun optional-keyword-sample-1 (a &optional b c &key d e)
  (list a b c d e))
(optional-keyword-sample-1 1)
(optional-keyword-sample-1 1 2)
(optional-keyword-sample-1 1 2 3)
(optional-keyword-sample-1 1 2 3 :e 5)
(optional-keyword-sample-1 1 :e 5)	; should not use &optional and &key

;; some structure to macro
(defmacro destructuring-sample-1 ((a b) (c d))
  `(list ',a ',b ',c ',d))
(destructuring-sample-1 (1 2) (3 4))
(destructuring-sample-1 ('mory 'ann) ('jenny 'moe))

;; nest is allow
(defmacro destructuring-sample-3 ((a &key b) (c (d e) &optional f))
  `(list ,a ,b ,c ,d ,e ,f))
(destructuring-sample-3 (1) (3 (4 5)))


;; extend example, should work on Mac Common Lisp
(defmacro with-processes ((name 
			   (pid num-processes)
			   (work-item work-queue)) &body body)
  (let ((process-fn (gensym))
	(items (gensym))
	(items-lock (gensym)))
    `(let ((,items (copy-list ,work-queue))
	   (,items-lock (make-lock)))
       (flet ((,process-fn (,pid)
		(let ((,work-item nil))
		  (loop 
		     (with-lock-grabbed (,items-lock)
		       (setq ,work-item (pop ,items)))
		     (when (null ,work-item)
		       (return))
		     ;; (format t "~&running id ~D~%" ,pid)
		     ,@body))))
	 (dotimes (i ,num-processes)
	   ;; (format t "~&creating id ~D~%" ,id)
	   (process-run-function
	    (format nil "~A-~D" ,name i)
	    #',process-fn
	    i))))))

(with-processes ("Test"
		 (id 3)
		 (item '(1 2 3 4 5 6 7 8 9 10 11 12 13)))
  (format t "~&id ~D item ~A~%" id item)
  (sleep (random 1.0)))