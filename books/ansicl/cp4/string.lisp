;;; string is vector of chars

; ascii 
(char-code #\a)
(code-char 92)

; compare
(sort "elbow" #'char<)

; select
(aref "abc" 1)
(char "abc" 1) ; faster

; example
(let ((str (copy-seq "Merlin")))
  (setf (char str 3) #\k)
  str)

(equal "fred" "fred")
(equal "fred" "Fred")
(string-equal "fred" "Fred")

; format
(format nil "~A or ~A" "truth" "dare")

; concat
(concatenate 'string "not " "to worry")

