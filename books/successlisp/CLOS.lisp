;; dispatch using defmethod
;; parameter list is a specialized list with the type of parameter.
(defmethod method1 ((param1 number) (param2 string)))
(defmethod method2 ((param1 float) (param2 sequence)))

;; if leave, match any data
(defmethod method3 (param1 (param2 vector)))

;; In CLOS, method is not a part of any class

;; more specific method is chosen when more than one lambda list match
(defmethod op2 ((x number) (y number)))
(defmethod op2 ((x integer) (x integer))) 
(op2 11 23)

;; 
(defmethod op3 ((x float) (y number))
(defmethod op3 ((x number) (y float)))
(op3 5.3 4.1)

(defmethod idiv ((numerator integer) (denominator integer))
  (values (floor numerator denominator)))
(defmethod idiv ((numberator integer) (denominator (eql 0)))
  nil)



;; Object inheritance 
(defclass c1 () nil)
(defclass c2 () nil)
(defclass c3 (c1) nil)
(defclass c4 (c2) nil)
(defclass c5 (c3 c2) nil)
(defclass c6 (c5 c1) nil)
(defclass c7 (c4 c3) nil)

(class-precedence-list (find-class 'c6))) ; not in clozure


;; Method combinations
(defmethod madness :before ())
(defmethod madness :after ())
(defmethod madness :around ())

;; primary method
(defmethod combol ((x number)) 
  (print 'primary) 1)
;; before method
(defmethod combol :before ((x integer)) 
  (print 'before-integer) 2)
(defmethod combol :before ((x rational))
  (print 'before-rational) 3)
;; after method
(defmethod combol :after ((x integer))
  (print 'after-integer) 4)
(defmethod combol :after ((x rational))
  (print 'after-rational) 5)

;; around method
;; strange
(defmethod combo1 :around ((x integer))
  (print 'around-float-before-call-next-method)
  (let ((result (call-next-method x)))
    (print 'around-float-after-call-next-method)
    result))