;;; feature expression
(defun foo ()
  #+allegro (do-one-thing)
  #+sbcl (do-another-thing)
  #+clisp (do-yet-another-thing)
  #-(or allegro sbcl clisp cmu) (error "Not implemented"))



