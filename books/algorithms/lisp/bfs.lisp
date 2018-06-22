; A - B   E 
;   \   \ | \
;    C    F - I
;     \     /
;      D - G

(defparameter bfs-G '((A B C)
                      (B A F)
                      (C A D)
                      (D C G)
                      (E F I)
                      (F B E I)
                      (G D I)
                      (I E F G)))

(defun graph-heads (graph)
  (mapcar #'first graph))

(defun graph-childrens (graph head)
  (labels ((filter (graph head)
	     (remove-if-not #'(lambda (x)
				(eql (first x) head))
			    graph)))
    (rest (first (filter graph head)))))

;;            A
;;          B   C
;;        F       D 
;;      E   I   G 
      
(defun bfs-traverse (graph start)
  (let ((ready nil)
        (completed nil))
    (push start ready)
    (dotimes (i (length graph))
      (when (not (null ready))
        (let* ((current (pop ready))
               (children (graph-childrens graph current)))
          (print current)
          (push current completed)
          (dolist (child children)
            (when (not (member child completed))
              (setf ready (append ready `(,child))))))))))

; A - B   E 
;   \   \ | 
;    C    F - I
;     \     
;      D - G


(defparameter dfs-G '((A B C)
                      (B A F)
                      (C A D)
                      (D C G)
                      (E F)
                      (F B E I)
                      (G D)
                      (I F)))


(defun dfs-traverse (graph start &optional (visited nil))
  (let ((children (graph-childrens graph start)))
    (print start)
    (push start visited)
    (print visited)
    (dolist (child children)
      (when (not (member child visited))
        (dfs-traverse graph child visited)))))

(defun dfs-traverse-2 (graph start)
  "avoid unnecessary repetion"
  (let ((visited nil))
    (labels ((helper (graph start)
               (let ((children (graph-childrens graph start)))
                 (print start)
                 (push start visited)
                 (print visited)
                 (dolist (child children)
                   (when (not (member child visited))
                     (helper graph child))))))
      (helper graph start))))
