(defvar family)
(setf family
      '((colin nil nil)
	(deirdre nil nil)
	(arthur nil nil)
	(kate nil nil)
	(frank nil nil)
	(linda nil nil)
	(suzanne colin deirdre)
	(bruce arthur kate)
	(charles arthur kate)
	(david arthur kate)
	(ellen arthur kate)
	(george frank linda)
	(hillary frank linda)
	(andre nil nil)
	(tamara bruce suzanne)
	(vincent bruce suzanne)
	(wanda nil nil)
	(ivan george ellen)
	(julie george ellen)
	(marie george ellen)
	(nigel andre hillary)
	(frederick nil tamara)
	(zelda vincent wanda)
	(joshua ivan wanda)
	(quentin nil nil)
	(robert quentin julie)
	(olivia nigel marie)
	(peter nigel marie)
	(erica nil nil)
	(yvette robert zelda)
	(diane peter erica)))

(defun father (name)
  (second (assoc name family)))

(defun mother (name)
  (third (assoc name family)))

(defun parents (name)
  (union (and (mother name) (list (mother name)))
	 (and (father name) (list (father name)))))

(defun children (name)
  (and name
       (mapcar #'first
	       (remove-if-not
		#'(lambda (x) (and (member name x)
				   (not (equal name (first x)))))
		family))))

  (defun siblings (name)
    (let ((children1 (children (father name)))
	  (children2 (children (mother name))))
      (set-difference 
       (union children1 children2)
       (list name))))

(defun mapunion (func list0)
  (reduce #'union (mapcar func list0)))

(defun grandparents (name)
  (mapunion #'parents
	    (parents name)))
  
(defun cousins (name)
  (mapunion #'children
	    (mapunion #'siblings
		      (parents name))))
  
(defun descended-from (name1 name2)
  (if (null name1)
      nil
      (or (member name2 (parents name1))
	  (descended-from (father name1) name2)
	  (descended-from (mother name1) name2))))

(defun ancestors (name)
  (if (null name)
      nil
      (append (parents name)
	      (ancestors (father name))
	      (ancestors (mother name)))))

(defun generation-gap (name1 name2)
  (generation-gap-helper name1 name2 0))

(defun generation-gap-helper (name1 name2 gap)
  (cond ((null name1) nil)
	((equal name1 name2) gap)
	(t (or (generation-gap-helper
		(father name1) name2 (1+ gap))
		(generation-gap-helper
		 (mother name1) name2 (1+ gap))))))
      
      
		       
