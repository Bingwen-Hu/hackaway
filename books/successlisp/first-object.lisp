(defclass empty-object () ())
(make-instance 'empty-object)
(find-class 'empty-object)
(make-instance (find-class 'empty-object))
;; note: make-instance create a new object every time
;; check an object: using EQ


;; objects have slots, with more options than structures.
(defclass 3d-point () (x y z))

;; class do not have accessor for attributes, should use SLOT-VALUE
;; try to get a slot before set it will cause an error
(let ((a-point (make-instance '3d-point)))
  (setf (slot-value a-point 'x) 0); set the X slot
  (slot-value a-point 'x))	  ; get the X slot

;; customize accessors of a class
(defclass 3d-point2 ()
  ((x :accessor point-x)
   (y :accessor point-y)
   (z :accessor point-z)))

;; make access convenient
(let ((a-point (make-instance '3d-point2)))
  (setf (point-x a-point) 0)
  (point-x a-point))

;; separate accessor names for reading and writing
(defclass 3d-point3 ()
  ((x :reader get-x :writer set-x)
   (y :reader get-y :writer set-y)
   (z :reader get-z :writer set-z)))

(let ((a-point (make-instance '3d-point3)))
  (set-z 3 a-point)
  (get-z a-point))



;; Override a slot accessor to do things that the client can't
(defclass sphere ()
  ((x :accessor x)
   (y :accessor y)
   (z :accessor z)
   (radius :accessor radius)
   (volume :reader volumn)
   (translate :writer translate)))


(defmethod volumn ((object sphere))
  (* 4/3 pi (expt (radius object) 3)))


;; Define classes with single inheritance for specialization
(defclass 2d-object () ())

(defclass 2d-centered-object (2d-object)
  ((x :accessor x)
   (y :accessor y)
   (orientation :accessor orientation)))


(defclass oval (2d-centered-object)
  ((axis-1 :accessor axis-1)
   (axis-2 :accessor axis-2)))

(defclass regular-polygon (2d-centered-object)
  ((n-sides :accessor number-of-sides)
   (size :accessor size)))

(setf rp (make-instance 'regular-polygon))
(setf (x rp) 1)
(setf (y rp) 2)
(setf (number-of-sides rp) 2)


;; options control initialization and provide documentation
; init
(defclass 3d-point4 ()
  ((x :accessor point-x :initform 0)
   (y :accessor point-y :initform 0)
   (z :accessor point-z :initform 0)))

; keyword init
(defclass 3d-point5 ()
  ((x :accessor point-x :initform 0 :initarg :x)
   (y :accessor point-y :initform 0 :initarg :y)
   (z :accessor point-z :initform 0 :initarg :z)))

(make-instance '3d-point5 :x 32 :y 17 :z -5)

; docs and type
(defclass 3d-point6 ()
  ((x :accessor point-x :initform 0 :initarg :x
      :documentation "x coordinate" :type real)
   (y :accessor point-y :initform 0 :initarg :y
      :documentation "y coordinate" :type real)
   (z :accessor point-z :initform 0 :initarg :z
      :documentation "z coordinate" :type real)))

