(defvar rooms nil)
(defvar loc nil)
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
