;; prim algorithms is one kind of mininum spanning tree algorithms
(defparameter G '((A (B . 6) (D . 4))
		  (B (A . 6) (D . 8) (C . 7) (E . 6))
		  (C (B . 7) (E . 4))
		  (D (A . 4) (B . 8) (E . 14) (F . 5))
		  (E (B . 6) (C . 4) (D . 14) (F . 7) (G . 8))
		  (F (D . 5) (E . 7) (G . 10))
		  (G (E . 8) (F . 10))))

(defun vertexs (graph)
  (mapcar #'first graph))


(defun adjacent (graph node)
  (rest 
   (first
    (remove-if-not #'(lambda (lst)
		      (eql (first lst) node))
		  graph))))

(defun adjacent-nodes (graph node)
  (let ((adj (adjacent graph node)))
    (mapcar #'first adj)))

(defun adjacent-weight (graph node)
  (let ((adj (adjacent graph node)))
    (mapcar #'rest adj)))

(defun nearest-node (graph node exclude)
  "return nearest node except node in exclude"
  (let ((adj (adjacent graph node)))
    (labels ((helper (min-adj adj)
               (cond ((null adj) (first min-adj))
                     ((< (rest min-adj) (rest (first adj)))
                      (helper min-adj (rest adj)))
                     (t (helper (first adj) (rest adj)))))))))

; step 1: choose a start node add to tree queue
; step 2: add adjacent nodes of 'the newest added tree node' to fringe queue
; step 3: choose the nearest node add to tree queue
; step 4: if all node in tree end else goto step 2
(defun prim (graph start)
  (let ((tree nil))
    (labels ((helper (current tree)
               (let ((mininum (nearest-node graph current tree)))
                 (when mininum 
                   (push mininum tree)
                   (helper mininum tree)))))
      (helper start tree))
    tree))
