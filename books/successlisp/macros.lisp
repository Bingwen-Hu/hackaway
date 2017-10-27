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
