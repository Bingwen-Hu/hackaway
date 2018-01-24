;;; vector
(vector)
(vector 1)
(vector 1 2)

;;; more generic make-array
(make-array 5 :initial-element nil)

;;; fill-pointer
(defparameter *x* (make-array 5 :fill-pointer 0))
(vector-push 'a *x*)
(vector-push 'b *x*)

;;; adjustable
(make-array 5 :fill-pointer 0 :adjustable t)


;;; vectors as sequences
(defparameter *y* (vector 1 2 3))

(length *y*)
(elt *y* 2)
(setf (elt *y* 2) 10)

(count 1 #(1 2 1 2 3 4 1 2 3 4))
(remove 1 #(1 1 2 3 1 2 1 2 3 4))
(remove 1 '(1 2 3 1 2 3 4 2 3 2))
(remove #\a "annMory")
(substitute 10 1 #(1 2 2 1 2 3 4 5 6 3))
(substitute 10 1 #(2 3 1 3 1 2 3 2 3 5))
(find 1 #(1 2 3 2 12 3 3 4))
(find 10 #(1 3 2 1 4 5 3 5 3))
(position 1 #(0 1 2 3 4 1 3 32 3 31))

(count "foo" #("foo" "bar" "baz") :test #'string=)
(find 'c #((a 10) (b 12) (c 30) (d 30)) :key #'first)


;;; higher-order function variants
(count-if #'evenp #(1 2 3 4 5))
(count-if-not #'evenp #(2 3 4 5 6))
(position-if #'digit-char-p "abcde0001")
(remove-if-not #'(lambda (x) (char= (elt x 0) #\f))
               #("foo" "bar" "baz" "foom"))

(concatenate 'vector #(1 2 3) '(3 4 5))
(concatenate 'list #(1 2 3) '(2 3 3))
(concatenate 'string "abc" '(#\d #\m #\a))

(sort (vector "foo" "bar" "baz") #'string<)
(merge 'vector #(1 3 5) #(2 4 6) #'<)
(merge 'list #(1 3 5) #(2 4 6) #'<)

(subseq "moryannjenny" 4 7)
(search "ann" "moryannjenny")
(mismatch "ann" "moryannjenny" :start2 4)


;;; sequence predicates
(every #'evenp #(1 2 3 4 5))
(some #'evenp #(1 2 3 4 5))
(notany #'evenp #(1 2 3 4 5))
(notevery #'evenp #(1 2 3 4 6))

(notevery #'> #(1 2 3 3) #(2 3 4))


;;; mapping
(map 'vector #'* #(1 2 3 4 5) #(10 9 8 7 6))
(let ((result '(1 2 8))
      (a '(1 2 3))
      (b '(4 5 6)))
  (map-into result #'+ result a b))



;;; hash table
(let ((h (make-hash-table)))
  (setf (gethash 'foo h) 'quux)
  (gethash 'foo h))

(defun show-value (key hash-table)
  (multiple-value-bind (value present) (gethash key hash-table)
    (if present
        (format nil "value ~a actually present." value)
        (format nil "value ~a key not found." value))))

(defun maphash-mory (hash-table)
  (maphash 
   #'(lambda (k v) (format t "~a => ~a~%" k v)) 
   hash-table))

(let ((hash-table (make-hash-table)))
  (setf (gethash 'name hash-table) 'mory)
  (setf (gethash 'like hash-table) 'Surpasser)
  (maphash-mory hash-table))

