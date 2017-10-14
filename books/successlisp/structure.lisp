(defstruct foo-struct a b c)

(let ((foo-1 (make-foo-struct :a 1 :b "two")))
  (print (foo-struct-b foo-1))
  (print (foo-struct-c foo-1))
  (values))                             ; suppress the return value


(defstruct ship 
  (name "unnamed")
  player
  (x-pos 0.0)
  (y-pos 0.0)
  (x-vel 0.0)
  (y-vel 0.0))


(make-ship :name "Excalibur" :player "Mory" :x-pos 100.0 :y-pos 221.0)

;;; using our own print-function, like __str__ or __repr__ in Python
(defstruct (ship 
             (:print-function
              (lambda (struct stream depth)
                (declare (ignore depth))
                (print-unreadable-object (struct stream)) ; add #<> at beginning
                (format stream "[ship ~A of ~A at (~D, ~D) moving (~D, ~D)]"
                        (ship-name struct)
                        (ship-player struct)
                        (ship-x-pos struct)
                        (ship-y-pos struct)
                        (ship-x-vel struct)
                        (ship-y-vel struct)))))
  (name "unnamed")
  player
  (x-pos 0.0)
  (y-pos 0.0)
  (x-vel 0.0)
  (y-vel 0.0))


;;; alter the way structures are stored in memory

(defstruct (bar 
             (:type vector))
  a b c)

;;; a more sweet to get the predicate
(defstruct (bar-2 
             (:type vector)
             :named)
  a b c)

(bar-2-p (make-bar-2))

;;; shorten slot accessor names
(defstruct (galaxy-class-cruiser-ship
             (:conc-name gcc-ship-))
  name 
  player
  (x-pos 0.0)
  (y-pos 0.0)
  (x-vel 0.0)
  (y-vel 0.0))

(let ((ship (make-galaxy-class-cruiser-ship)))
  (print (gcc-ship-x-pos ship))
  (values))


;;; allocate new structures without using keywords

; Note that the make-3d-point is disappear and default value is gone
(defstruct (3d-point
             (:constructor
              create-3d-point (x y z)))
  (x 1) (y 2) (z 3))

(create-3d-point 1 -2 3)


;;; Define one structure as an extension of another
(defstruct employee 
  name department salary social-security-number telephone)

(make-employee)

;;; inheritance
(defstruct (manager 
             (:include employee))
  bonus direct-reports)

(make-manager)

(employee-p (make-manager))             ;=> T
