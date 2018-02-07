;;; assume there is a class name bank-account and two subclass
;;; named checking-account and savings-account
(defgeneric withdraw (account amount)
  (:documentation "withdraw the specified amount from the account.
Signal an error if the current balance is less than amount."))

(defmethod withdraw ((account bank-account) amount)
  (when (< (balance account) amount)    ;balance is a property of account
    (error "Account overdrawn."))
  (decf (balance account) amount))


(defmethod withdraw ((account checking-account) amount)
  (let ((overdraft (- amount (balance account))))
    (when (plusp overdraft)
      (withdraw (overdraft-account account) overdraft)
      (incf (balance account) overdraft)))
  (call-next-method))
