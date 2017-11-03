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