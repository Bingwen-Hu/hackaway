(defvar database nil)
(setf database
      '((b1 shape brick)
	(b1 color green)
	(b1 size small)
	(b1 supported-by b2)
	(b1 supported-by b3)
	(b2 shape brick)
	(b2 color red)
	(b2 size small)
	(b2 supports b1)
	(b2 left-of b3)
	(b3 shape brick)
	(b3 color red)
	(b3 size small)
	(b3 supports b1)
	(b3 right-of b2)
	(b4 shape pyramid)
	(b4 color blue)
	(b4 size large)
	(b4 supported-by b5)
	(b5 shape cube)
	(b5 size large)
	(b5 color green)
	(b5 supports b4)
	(b6 shape brick)
	(b6 color purple)
	(b6 size large)))

(defun match-element (symbol1 symbol2)
  (cond ((equal symbol1 symbol2) t)
	((equal '? symbol2) t)
	(t nil)))

(defun match-triple (assertion pattern)
  (every #'match-element assertion pattern))

(defun fetch (pattern)
  (remove-if-not
   #'(lambda (x) (match-triple x pattern))
   database))

(defun block-color (bloc)
  (cons bloc '(color ?)))

(defun supporters (bloc)
  (mapcar #'first
	  (fetch (list '? 'supports block))))

  
(defun supp-cube (bloc)
  (member 'cube
	  (mapcar #'(lambda (b)
		      (third (first (fetch (list b 'shape '?)))))
		  (supporters bloc))))
	  
	 
(defun desc1 (bloc)			;using fetch make life easier
  (remove-if-not
   #'(lambda (x) (equal bloc (first x)))
   database))

(defun desc2 (bloc)
  (mapcar #'cdr (desc1 bloc)))

(defun description (bloc)
  (reduce #'append (desc2 bloc)))
