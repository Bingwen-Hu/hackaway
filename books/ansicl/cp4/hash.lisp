(defvar hash)
(setf hash (make-hash-table))


(setf (gethash 'color hash) 42)
(gethash 'color hash)

(setf (gethash 'color hash) nil)
(gethash 'color hash)
; remove entry
(remhash 'color hash)


(setf (gethash 'name hash) 'Mory
      (gethash 'age hash) 42)

(maphash #'(lambda (k v)
             (format t "~A => ~A~%" k v))
         hash)
