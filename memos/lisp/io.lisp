(defun space-over (n)
  (if (< n 0)
      (format t "~&error!")
      (dotimes (i n)
	(format t " "))))

(defun test (n)
  (format t "~&>>>")
  (space-over n)
  (format t "<<<"))

(defun plot-one-point (plotting-string y-val)
  (space-over y-val)
  (format t plotting-string)
  (format t "~%"))

(defun plot-points (plotting-string yv-list)
  (dolist (y-val yv-list)
    (plot-one-point plotting-string y-val)))

(defun generate (M N)
  (cond ((equal M N) (list N))
	(t (cons M (generate (1+ M) N)))))

(defun make-graph ()
  (let* ((func (prompt-for "function to graph? "))
	 (start (prompt-for "starting x value? "))
	 (end (prompt-for "ending x value? "))
	 (plotting-string (prompt-for "plotting-string? ")))
    (plot-points plotting-string
		 (mapcar func (generate start end)))
    t))

(defun prompt-for (prompt-string)
  (format t "~A" prompt-string)
  (read))
  
