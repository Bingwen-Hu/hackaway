;;; Behaviors are associdated with a class by defining generic functions and 
;;; methods specialized on the class. DEFCLASS only defines the class as a 
;;; data type.

; three facets of a class
; class name
; class relationship with other class, defaults: standard-object
; class member

(defclass bank-account () 
  (customer-name
   balance))

(defparameter account (make-instance 'bank-account))
(setf (slot-value account 'customer-name) "Mory")
(setf (slot-value account 'balance) 100000000)
(slot-value account 'balance)
(slot-value account 'customer-name)


;;; object initialization
(defclass bank-account ()
  ((customer-name
    :initarg :customer-name)
   (balance
    :initarg :balance
    :initform 0)))

(defparameter account 
  (make-instance 'bank-account 
                 :customer-name "Mory"
                 :balance 2000000))


(defvar *account-numbers* 0)
(defclass bank-account () 
  ((customer-name
    :initarg :customer-name
    :initform (error "Must supply a customer name."))
   (balance 
    :initarg :balance
    :initform 0)
   (account-number
    :initform (incf *account-numbers*))))


;;; want to define some property after user create the instance
;;; using a :after function
(defclass bank-account ()
  ((customer-name
    :initarg :customer-name
    :initform (error "Must supply a customer name."))
   (balance
    :initarg :balance
    :initform 0)
   (account-number
    :initform (incf *account-numbers*))
   account-type))

(defmethod initialize-instance :after ((account bank-account) &key)
  (let ((balance (slot-value account 'balance)))
    (setf (slot-value account 'account-type)
          (cond 
            ((>= balance 100000) :gold)
            ((>= balance 40000) :silver)
            (t :bronze)))))


;;; supply rich parameter to make-instance to modify the class behavior
(defmethod initialize-instance :after ((account bank-account)
                                       &key opening-bonus-percentage)
  (when opening-bonus-percentage
    (incf (slot-value account 'balance)
          (* (slot-value account 'balance) 
             (/ opening-bonus-percentage 100)))))

(defparameter account 
  (make-instance 'bank-account :customer-name "Ann Gewara"
                 :balance 1000 :opening-bonus-percentage 5))
(slot-value account 'balance)           ;=> 1050

;;; accessor
;;; reader: read only method
;;; writer: setf function
;;; accessor: both read and write
;;; documentation: docs....
(defclass bank-account ()
  ((customer-name 
    :initarg :customer-name
    :initform (error "Must supply a customer name.")
    :accessor customer-name
    :documentation "Customer's name")
   (balance 
    :initarg :balance
    :initform 0
    :reader balance
    :writer (setf balance)
    :documentation "Current account balance")
   (account-number
    :initform (incf *account-numbers*)
    :reader account-number
    :documentation "Account number, unique within a bank")
   (account-type
    :reader account-type
    :documentation "Type of account, one of :gold, :silver, or :bronze.")))

;;; two helper macro, WITH-SLOT and WITH-ACCESSOR
(defparameter *minimum-balance* 10000)
(defmethod assess-low-balance-penalty ((account bank-account))
  (with-slots (balance) account
    (when (< balance *minimum-balance*)
      (decf balance (* balance .01)))))

; or alias form
(defmethod assess-low-balance-penalty ((account bank-account))
  (with-slots ((bal balance)) account
    (when (< bal *minimum-balance*)
      (decf bal (* bal .01)))))

;;; only setup writable, this function will work correctly.
(defmethod assess-low-balance-penalty ((account bank-account))
  (with-accessors ((bal balance)) account
    (when (< bal *minimum-balance*)
      (decf bal (* bal .01)))))


;;; nested accessors
(defmethod merge-accounts ((account1 bank-account) (account2 bank-account))
  (with-accessors ((balance1 balance)) account1
    (with-accessors ((balance2 balance)) account2
      (incf balance1 balance2)
      (setf balance2 0))))
