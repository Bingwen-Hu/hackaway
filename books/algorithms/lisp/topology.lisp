;;; say we hava a graph
;;; 
;;;         C 
;;;       /   \
;;; A - B ----- E -- F  
;;;       \  /
;;;   G - D

;;; topology sort
(defparameter G '((A B)
                  (B C D E)
                  (C E)
                  (D E)
                  (E F)
                  (F)
                  (G D)))

(defun in-degree (graph node)
  (length 
   (remove-if-not #'(lambda (lst) 
                      (member node (rest lst)))
                  graph)))


(defun topology-sort (graph)
  )
