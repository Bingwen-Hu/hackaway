;;; say we have a graph
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
  (count-if #'(lambda (lst) 
		(member node (rest lst)))
	    graph))

(defun headnodes (graph)
  (mapcar #'first graph))

(defun narrow-graph (graph nodes)
  "remove nodes from graph, not really"
  (remove-if #'(lambda (lst)
		 (member (first lst) nodes))
	     graph))

;; zero-in-degree
(defun zero-in-degree (graph)
  "return nodes that is zero in degree"
  (let ((nodes (headnodes graph)))
    (remove-if-not #'(lambda (node)
		       (when (= 0 (in-degree graph node))
			 node))
		   nodes)))
    

(defun terminal-node (graph)
  (let ((terminal 
	 (remove-if-not #'(lambda (lst)
			    (eql nil (rest lst)))
			graph)))
    (first (first terminal))))



(defun topology-sort (graph)
  (let ((queue nil)
	(terminal (terminal-node graph)))
    (labels ((helper (q)
	       (setf q (append q (zero-in-degree (narrow-graph graph q))))
	       (if (member terminal q)
		   q
		   (helper q))))
      (helper queue))))
	

;; another terminal: zero-in-degree return nil

(defun topology-sort-2 (graph)
  (let ((queue nil))
    (labels ((helper (q)
	       (let ((zero-nodes (zero-in-degree (narrow-graph graph q))))
		 (if zero-nodes
		     (helper (append q zero-nodes))
		     q))))
      (helper queue))))

