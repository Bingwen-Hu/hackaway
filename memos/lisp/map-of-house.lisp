(setf rooms '((living-room
	       (north front-stairs)
	       (south dining-room)
	       (east kitchen))
	      (upstairs-bedroom
	       (south front-stairs)
	       (west library))
	      (dining-room
	       (north living-room)
	       (east pantry)
	       (west downstairs-bedroom))
	      (kitchen
	       (west living-room)
	       (south pantry))
	      (pantry
	       (west dining-room)
	       (north kitchen))
	      (downstairs-bedroom
	       (north back-stairs)
	       (east dining-room))
	      (back-stairs
	       (north library)
	       (south downstairs-bedroom))
	      (front-stairs
	       (upstairs-bedroom)
	       (south living-room))
	      (library
	       (east upstairs-bedroom)
	       (south back-stairs))))

(defun choices (room)
  (cdr (assoc room rooms)))

(defun look (direction origin)
  (cadr (assoc direction (choices origin))))
	      
(defun set-robbie-location (place)
  "Moves Robbie to PLACE by setting the variable LOC."
  (setf loc place))

(defun how-many-choices (place)
  (length (choices place)))

(defun upstairsp (place)
  (when (member place '(library upstairs-bedroom))
      T))

(defun onstairsp (place)
  (when (member place '(front-stairs back-stairs))
    T))

(defun where ()
  (cond ((upstairsp loc) (list 'Robbie 'is 'upstairs 'in 'the loc))
	((onstairsp loc) (list 'Robbie 'is 'on 'the loc))
	(t (list 'Robbie 'is 'downstairs 'in 'the loc))))

(defun move (direction)
  (let ((newloc (look direction loc)))
    (if newloc
	(progn
	  (set-robbie-location newloc)
	  (where))
	'(Ouch! Robbie  hit a wall!))))
