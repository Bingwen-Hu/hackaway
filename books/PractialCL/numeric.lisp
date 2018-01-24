(+ 1 2)
(+ 10.0 3.0)
(+ #c(1 2) #c(3 4))                     ;complex number
(/ 10 5 2)
(/ 2 3)
(/ 3)

;;; convert
(+ 1 2.0)
(/ 2 3.0)
(+ #c(1 2) 3)
(+ #c(1 2) 3/2)
(+ #c(1 1) #c(2 -1))


;;; function
(defvar x 0)
(incf x)
(incf x 10)
(decf x)
(decf x 5)


;;; equality
(= 1 1)
(= 10 20/2)
(= 1 1.0 #c(1.0 0.0) #c(1 0))


;;; inequality
(/= 1 1)
(/= 1 2)
(/= 1 2 3)
(/= 1 2 3 1)
(/= 1 2 3 1.0)


(max 10 11)
(min 10 2 3)

(zerop 0)
(zerop 1)
(minusp 1)
(minusp -1)
(plusp 1)
(plusp -1)


;;; character
; just note for some functions:
; char= char/= char< char> char<= char>=
; char-equal char-not-equal char-lessp char-greaterp 
; char-not-greaterp char-not-lessp

;;; string, similar to character
; string= string/= string< string> string<= string>=
; string-equal string-not-equal string-lessp string-greaterp 
; string-not-greaterp string-not-lessp 

(string= "mory" "mory")
(string> "mory" "Ann")
(string/= "lisp" "lissome")
(string< "moryann" "Jenny" :start1 3 :start2 2)
