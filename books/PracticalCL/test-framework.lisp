;;; Mory has nothing to do, then Mory begins to do something.
; basic test
(defun test-+ ()
  (and 
   (= (+ 1 2) 3)
   (= (+ 1 2 3) 6)
   (= (+ -1 -3) -4)))


(defun test-+2 ()
  (format t "~:[FAIL~;pass~] ... ~a~%" (= (+ 1 2) 3) '(= (+ 1 2) 3))
  (format t "~:[FAIL~;pass~] ... ~a~%" (= (+ 1 2 3) 6) '(= (+ 1 2 3) 6))
  (format t "~:[FAIL~;pass~] ... ~a~%" (= (+ -1 -2) -3) '(= (+ -1 -2) -3)))

; factor 1
(defun report-result (result form)
  (format t "~:[FAIL~;PASS~] ... ~a~%" result form))

(defun test-+3 ()
  (report-result (= (+ 1 2) 3) '(= (+ 1 2) 3))
  (report-result (= (+ 1 2 3) 6) '(= (+ 1 2 3) 6))
  (report-result (= (+ -1 -3) -4) '(= (+ -1 -3) -4)))


; code as data
; when you say (check (= (+ 1 2) 3))
; really means (report-result (= (+ 1 2) 3) '(= (+ 1 2) 3))
; then you need a macro

(defmacro check (form)
  `(report-result ,form ',form))


(defun test-+4 ()
  (check (= (+ 1 2) 3))
  (check (= (+ 1 2 3) 6))
  (check (= (+ -1 -3) -4)))

(defmacro check2 (&body forms)
  `(progn 
     ,@(loop for f in forms 
          collect `(report-result ,f ',f))))

; finally
(defun test-+5 ()
  (check2
   (= (+ 1 2) 3)
   (= (+ 1 2 4) 7) 
   (= (+ -1 -3) -4)))


; fixiing the return value

; macro define in macro.lisp
(defmacro with-gensyms ((&rest names) &body body)
  `(let ,(loop for n in names collect `(,n (gensym)))
     ,@body))


(defun report-result2 (result form)
  (format t "~:[FAIL~;PASS~] ... ~a~%" result form)
  result)

(defmacro combine-results (&body forms)
  (with-gensyms (result)
    `(let ((,result t))
       ,@(loop for f in forms
              collect `(unless ,f (setf ,result nil)))
       ,result)))


(defmacro check3 (&body forms)
  `(combine-results 
    ,@(loop for f in forms collect `(report-result2 ,f ',f))))


(defun test-+6 ()
  (check3 
   (= (+ 1 2) 3)
   (= (+ 1 2 3) 6)
   (= (+ -1 -2) -3)))


; better result reporting
(defun test-* ()
  (check3
   (= (* 2 2) 4)
   (= (* 3 5) 15)))

(defun test-arithmetic ()
  (combine-results
   (test-+6)
   (test-*)))

;;; problem, how can I know which case fail?
(defvar *test-name* nil)

(defun report-result3 (result form)
  (format t "~:[FAIL~;PASS~] ... ~a: ~a~%" result *test-name* form)
  result)

(defmacro check4 (&body forms)
  `(combine-results 
    ,@(loop for f in forms collect `(report-result3 ,f ',f))))


(defun test-+7 ()
  (let ((*test-name* 'test-+))
    (check4
     (= (+ 1 2) 3)
     (= (+ 1 2 3) 6)
     (= (+ -1 -3) -4))))


(defun test-*2 ()
  (let ((*test-name* 'test-*))
    (check4 
     (= (* 2 2) 4)
     (= (* 3 5) 15))))


(defun test-arithmetic2 ()
  (combine-results
   (test-+7)
   (test-*2)))

;;; half abstract 
;; so that's why I decide to learn lisp
(defmacro deftest (name parameters &body body)
  `(defun ,name ,parameters
     (let ((*test-name* (append *test-name* (list ',name))))
       ,@body)))

(deftest test-+8 ()
  (check4 
   (= (+ 1 2) 3)
   (= (+ 1 2 3) 6)
   (= (+ -1 -2) -3)))

(deftest test-*3 ()
  (check4 
    (= (* 1 2) 2)
    (= (* 3 5) 12)))

(deftest test-arithmetic3 ()
  (combine-results
    (test-+8)
    (test-*3)))


(deftest test-math ()
  (test-arithmetic3))
