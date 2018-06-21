(defparameter table (make-hash-table))

(setf (gethash 'AL table) 'Alabama)
(setf (gethash 'Ak table) 'Alaska)
(setf (gethash 'AZ table) 'Arizona)
(setf (gethash 'AR table) 'Arkansas)

(gethash 'AK table)
(remhash 'AZ table)
(gethash 'AZ table)