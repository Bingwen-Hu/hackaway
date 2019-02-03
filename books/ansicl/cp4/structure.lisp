;; simple accessor
(defun block-height (b)
  (svref b 0))


(defstruct point 
  x 
  y)

(defvar p)
(setf p (make-point :x 0 :y 0))
(setf (point-x p) 100)
(setf (point-y p) 42)
; literal
(type-of p)
(typep p 'point)

; with default value
(defstruct point 
  (x 100)
  (y 42))

; custom print function
(defstruct (point (:print-function print-point))
  (x 100)
  (y 42))

(defun print-point (p stream -)
  (format stream "#<~A,~A>" (point-x p) (point-y p)))

; custom accessor prefix
(defstruct (point (:conc-name p))
  (x 0)
  (y 42))
