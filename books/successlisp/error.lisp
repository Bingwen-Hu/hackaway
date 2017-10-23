;; unwind-protect
(let (resource stream)
  (unwind-protect 
       (progn 
	 (setq resource (allocate-resource)
	       stream (open-file))
	 (process stream resource))
    (when stream (close stream))
    (when resource (deallocate resource))))

;; gracious exit with BLOCK and RETURN-FROM
(defun block-demo (flag)
  (print 'before-outer)
  (block outer 
    (print 'before-inner)
    (print (block inner
	     (if flag
		 (return-from outer 7)
		 (return-from inner 3))
	     (print 'never-print-this)))
    (print 'after-inner)
    t))
	 
