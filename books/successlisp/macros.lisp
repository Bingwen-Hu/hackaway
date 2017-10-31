;; macro return forms instead of values

;; Backquote
`(the sum of 17 and 83 is ,(+ 17 93))

(defmacro swap (a b)
  `(let ((temp ,a))
     (setf ,a ,b)
     (setf ,b temp)))


(let ((x 3)
      (y 7))
  (swap x y)
  (list x y))

(pprint (macroexpand-1 '(swap x y)))

;; macro as functions
(defmacro sortf (place)
  `(setf ,place (sort ,place)))

(defmacro togglef (place)
  `(setf ,place (not ,place))

(defmacro either (form1 form2)
  `(if (zerop (random 2)) ,form1 ,form2))



;; beyond the obvious part 1
;; create a lookup table in compile time and look up value from it in runtime
(defvar *sin-tables* (make-hash-table)
  "A hash table of tables of sine values. The hash is keyed by the number of 
entries in each sine table.")

;; only used in compile time
(defun get-sin-table-and-increment (divisions)
  "Returns a sine lookup table and the number of radians quantized by each
entry in the table. Tables of a given size are reused. A table covers angles
from zero to pi/4 radians."
  (let ((table (gethash divisions *sin-tables* :none))
	(increment (/ pi 2 divisions)))
    (when (eq table :none)
      (setq table
	    (setf (gethash divisions *sin-tables*)
		  (make-array (1+ divisions) :initial-element 1.0)))
      (dotimes (i divisions)
	(setf (aref table i)
	      (sin (* increment i)))))
    (values table increment)))

(defmacro lookup-sin (radians divisions)
  "Return a sine value via table lookup."
  (multiple-value-bind (table increment)
      (get-sin-table-and-increment divisions)
    `(aref ,table (round ,radians ,increment))))
      
      
	   

;; beyond the obvious part 2: macros define macros
(defmacro defsynonym (old-name new-name)
  "Define OLD-NAME to be equivalent to NEW-Name when used in the first position
of a Lisp form."
  `(defmacro ,new-name (&rest args)
     `(,',old-name ,@args)))

;; tricks of trade: elude capture using GENSYM
(defmacro repeat (times &body body)
  `(dotimes (x ,times)
     ,@body))

;; a better version
(defmacro repeat (times &body body)
  (let ((x (gensym)))
    `(dotimes (,x ,times)
       ,@body)))