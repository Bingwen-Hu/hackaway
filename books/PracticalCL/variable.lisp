;;; let form and lexical variables
(defun foo (x)
  (format t "Parameter: ~a~%" x)
  (let ((x 2))
    (format t "let param: ~a~%" x)))


;;; defvar 
;;; defparameter
;;; defconstant

;;; setf
