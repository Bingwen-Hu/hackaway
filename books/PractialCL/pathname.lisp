;;; pathname is special in Common Lisp
(defparameter mory-dir "/home/mory/hackaway/ideas.md")
(let* ((pathname-obj (pathname mory-dir))
       (directory (pathname-directory pathname-obj))
       (name (pathname-name pathname-obj))
       (type (pathname-type pathname-obj)))
  (values directory name type))

;;; #p"string" -> (pathname "string")
(namestring #p"/home/mory/hackaway/ideas.md")
(directory-namestring #p"/home/mory/hackaway/ideas.md")
(file-namestring #p"/home/mory/hackaway/ideas.md")


;;; make-pathname create a pathname object
(make-pathname
 :directory '(:absolute "home" "mory" "hackaway")
 :name "ideas"
 :type "md")

;;; robust pathname
(make-pathname 
 :name "YouKnowMe"
 :type "py" 
 :defaults #p"/home/mory/hackaway/ideas.md")

(make-pathname
 :name "Relative"
 :type "lisp"
 :directory '(:relative "books")
 :defaults #p"/home/mory/hackaway/ideas.md")


;;; intuitive
(merge-pathnames #p"/home/mory/hackaway/" #p"ideas.md")
;;; lack of a '/' is wrong, so a pathname library is needed.
(merge-pathnames #p"/home/mory/hackaway" #p"ideas.md")
;;; switch position is also works
(merge-pathnames #p"ideas.md" #p"/home/mory/hackaway/")
;;; get directory relative to root directory
(enough-namestring #p"/home/mory/hackaway" #p"/home/")


(with-open-file (in mory-dir :element-type '(unsigned-byte 8))
  (file-length in))
