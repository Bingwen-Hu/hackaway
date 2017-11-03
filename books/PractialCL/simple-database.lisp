(defun make-cd (title artist rating ripped)
  (list :title title :artist artist :rating rating :ripped ripped))

(defvar *db* nil)			;only initial once

;; a little abstraction
(defun add-record (cd)
  (push cd *db*))

;; usage
(add-record (make-cd "Rose" "Kathy Mattea" 7 t))
(add-record (make-cd "Fly" "Diie Chicks" 8 t))

;; beautiful print
(defun dump-db ()
  (dolist (cd *db*)
    (format t "~{~a:~10t~a~%~}~%" cd)))

;; little bit: FORMAT loop
;; begin with ~{ and end with ~}
(format t "~{~A: ~A~%~}" '(Name Mory Love Jenny Belief Buddism))

;; Inprove user interface
;; *query-io* is a global stream
(defun prompt-read (prompt)
  (format *query-io* "~a: " prompt)
  (force-output *query-io*)
  (read-line *query-io*))

;; not very good
(defun prompt-for-cd-not-good ()
  (make-cd 
   (prompt-read "Title")
   (prompt-read "Artist")
   (prompt-read "Rating")
   (prompt-read "Ripped [y/n]")))
		
;; better
(defun prompt-for-cd ()
  (make-cd
   (prompt-read "Title")
   (prompt-read "Artist")
   (or (parse-integer (prompt-read "Rating") :junk-allowed t) 0)
   (y-or-n-p "Ripped [y/n]: ")))

;; wrap up
(defun add-cds ()
  (loop (add-record (prompt-for-cd))
     (if (not (y-or-n-p "Another? [y/n]: ")) (return))))

;; save and loading the database
(defun save-db (filename)
  (with-open-file (out filename
		       :direction :output
		       :if-exists :supersede)
    (with-standard-io-syntax		;ensure syntax is standard
      (print *db* out))))		;print lisp form out

(defun load-db (filename)
  (with-open-file (in filename)
    (with-standard-io-syntax
      (setf *db* (read in)))))


;; define query functions
(defun select-by-artist (artist)
  (remove-if-not 
   #'(lambda (cd) 
       (equal (getf cd :artist) artist))
   *db*))

;; in order to abstract the artist out of the function Select-By-Artist
;; we define a common selector
(defun select (selector-fn)
  (remove-if-not selector-fn *db*))

(defun artist-selector (artist)
  #'(lambda (cd) (equal (getf cd :artist) artist)))

(select (artist-selector "Mory"))


;; very important note:
;; WHERE return a function as a selector
(defun where (&key title artist rating (ripped nil ripped-p))
  #'(lambda (cd)
      (and
       (if title    (equal (getf cd :title)  title)  t)
       (if artist   (equal (getf cd :artist) artist) t)
       (if rating   (equal (getf cd :rating) rating) t)
       (if ripped-p (equal (getf cd :ripped) ripped) t))))


;; update
(defun update (selector-fn &key title artist rating (ripped nil ripped-p))
  (setf *db*
	(mapcar 
	 #'(lambda (row)
	     (when (funcall selector-fn row) ;exist
	       (if title    (setf (getf row :title)  title))
	       (if artist   (setf (getf row :artist) artist))
	       (if rating   (setf (getf row :rating) rating))
	       (if ripped-p (setf (getf row :ripped) ripped)))
	     row) 			;return value of each map
	 *db*)))			;totally reset *db*



(defun delete-rows (selector-fn)
  (setf *db* (remove-if selector-fn *db*)))

