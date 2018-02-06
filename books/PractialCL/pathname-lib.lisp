;;; feature expression
(defun foo ()
  #+allegro (do-one-thing)
  #+sbcl (do-another-thing)
  #+clisp (do-yet-another-thing)
  #-(or allegro sbcl clisp cmu) (error "Not implemented"))


(defun component-present-p (value)
  "Test whether a given component of a pathname is 'present'"
  (and value (not (eql value :unspecific))))

(defun directory-pathname-p (p)
  "test whether a pathname is already in directory form"
  (and 
   (not (component-present-p (pathname-name p)))
   (not (component-present-p (pathname-type p)))
   p))

(defun pathname-as-directory (name)
  (let ((pathname (pathname name)))
    (when (wild-pathname-p pathname)
      (error "Can't reliably convert wild pathnames."))
    (if (not (directory-pathname-p name))
        (make-pathname
         :directory (append (or (pathname-directory pathname)
                                (list :relative))
                            (list (file-namestring pathname)))
         :name nil
         :type nil
         :defaults pathname)
        pathname)))



;;; CLisp special
(defun directory-wildcard (dirname)
  (make-pathname
   :name :wild
   :type #-clisp :wild #+clisp nil
   :defaults (pathname-as-directory dirname)))


(defun list-directory (dirname)
  (when (wild-pathname-p dirname)
    (error "Can only list concrete directory names."))
  (let ((wildcard (directory-wildcard dirname)))
    
    #+(or sbcl cmu lispworks)
    (directory wildcard)
    
    #+ppenmcl
    (directory wildcard :directories t)
    
    #+allegro
    (directory wildcard :directories-are-files nil)
    
    #+clisp
    (nconc
     (directory wildcard)
     (directory (clisp-subdirectories-wildcard wildcard)))
    
    #-(or sbcl cmu lispworks openmcl allegro clisp)
    (error "list-directory not implemented")))

#+clisp
(defun clisp-subdirectories-wildcard (wildcard)
  (make-pathname
   :directory (append (pathname-directory wildcard) (list :wild))
   :name nil
   :type nil
   :defaults wildcard))
