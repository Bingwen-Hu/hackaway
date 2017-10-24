;; concatenate: new sequences from old
;; vector and list are both sequences
(concatenate 'list)
(concatenate 'vector)
(concatenate 'list '(1 2 3) '(4 5))
(concatenate 'vector '(1 2 3) '(5 6))
(concatenate 'vector '(1 2 3) #(4 9))
(concatenate 'list "Hello")

;; ELT and SUBSEQ get what you want
;; they are less specific and less efficient

(elt '(1 2 3 4 5) 1)
(subseq '(1 2 3 4 5) 2)
(subseq '(1 2 3 4 5 5) 2 5)
(copy-seq '(a b c))

;; reverse and nreverse
(defun collect-even-numbers (number-list)
  (let ((result ()))
    (dolist (numer number-list)
      (when (evenp number)
	(push number result)))
    (nreverse result)))

;; length ...
;; count : when it's what's inside that matters
(count 3 '(1 2 3 3 3 4 5 6 7 8 8 8 8))
(count-if #'oddp '(1 3 3 34 5 5 4 3 3 4 3 3))
(count-if-not #'evenp '(1 3 3 4 5 3 4 5 6))

;; keyword arguments there functions accept
;; start . end . from-end . key
(count 3 '((1 2 3) (2 3 1) (3 1 2) (2 5 4) (2 3 4)) :key #'second)
(count 3 '(2 1 3 4 5 6 4 6) :test #'<)

;; remove, substitute and other sequence changers
(remove 7 '(1 2 3 a bc d nil 9 9 3 7 7 7))
(substitute '(q) 7 '(1 2 3 7 7 7 3 d a b e))

(remove-duplicates '(1 2 3 4 5 6 (1 2 3) f c g c h e i j l b a c e f s F G H I))
		   
;; fill and replace  Not very useful :-)
(fill (list 1 2 3 4 5 7) 8)
(replace '(1 2 nil 3 4 5 nil 3 9) 
	 '(1 2 3 4 5 6 7 8 9))


;; Position, find, search, and mismatch
(position #\a "This is all about you!")
(find #\a "This is all about you, isn't it?")
(search "ab" "this is all about you!")
(mismatch "banana" "bananananono")
(mismatch "." "...hello")

;; SORT and MERGE
(sort (list 1 3 4 5 6 7 7 5 4 3) #'>)
(stable-sort '(1 3 4 56 5 4 3 34 5 4 3 4) #'>)

(merge 'vector '(1 3 5 9 8) #(2 6 4 7 0) #'>)