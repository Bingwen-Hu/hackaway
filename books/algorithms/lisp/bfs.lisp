; A - B   E 
;   \   \ | \
;    C    F - I
     \     /
;      D - G

(defparameter G '((A B C)
		  (B A F)
		  (C A D)
		  (D C G)
		  (E F I)
		  (F B E I)
		  (G D I)
		  (I E F G)))

(defvar completed '())
(defvar ready '())

(defun graph-heads (graph)
  (mapcar #'first graph))

(defun graph-childrens (graph head)
  (labels ((filter (graph head)
	     (remove-if-not #'(lambda (x)
				(eql (first x) head))
			    graph)))
    (rest (first (filter graph head)))))

;;