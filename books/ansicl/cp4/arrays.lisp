; arrays in lisp
(let ((arr (make-array '(2 3) :initial-element nil)))
  (aref arr 0 1))


; one dimension array -- vector
(vector 'a "d" 3)
; literal
#("a" 'b 3)

(svref #("a" b e) 0)
; sv => simple vector

;;; example: binary search
(defun bin-search (obj vec)
  (let ((len (length vec)))
    (and (not (zerop len))
         (finder obj vec 0 (- len 1)))))


(defun finder (obj vec start end)
  "seems meaningless"
  (format t "~A~%" (subseq vec start (+ end 1)))
  (let ((range (- end start)))
    (if (zerop range)
        (if (eql obj (aref vec start))
            obj
            nil)
        (let ((mid (+ start (round (/ range 2)))))
          (let ((obj2 (aref vec mid)))
            (if (< obj obj2)
                (finder obj vec start (- mid 1))
                (if (> obj obj2)
                    (finder obj vec (+ mid 1) end)
                    obj)))))))
