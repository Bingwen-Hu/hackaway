;;;; Found a unit test framework step by step
;;;; Abstraction is essential

;; Problem 1: want to test something like these
(= (+ 1 2) 3)
(= (+ 1 2 3) 6)
(= (+ 1 -3) -2)

;; STEP 1: Just write a function
(defun test-+% ()
  (and
   (= (+ 1 2) 3)
   (= (+ 1 2 3) 6)
   (= (+ 1 -3) -2)))

;; easily use
(test-+%)

;; Problem 2: how can I know which test fail?
;; STEP 2: print more!
(defun test-+%% ()
  (format t "~:[FAIL~;pass~] ... ~a~%" (= (+ 1 2) 3) '(= (+ 1 2) 3))
  (format t "~:[FAIL~;pass~] ... ~a~%" (= (+ 1 2 3) 6) '(= (+ 1 2 3) 6))
  (format t "~:[FAIL~;pass~] ... ~a~%" (= (+ 1 -3) -2) '(= (+ 1 -3) -2)))

;; Problem 3: No! too much repetition! too long!
;; STEP 3: wrap up the print form
(defun report-result (result form)
  (format t "~:[FAIL~;pass~] ... ~a~%" result form))

(defun test-+%%% ()
  (report-result (= (+ 1 2) 3) '(= (+ 1 2) 3))
  (report-result (= (+ 1 2 3) 6) '(= (+ 1 2 3) 6))
  (report-result (= (+ 1 -3) -2) '(= (+ 1 -3) -2)))

;; now looks better. What's about the repetive input form?
;; Problem 4: repetive means mistakes!
;; STEP 4: Just need a macro!
(defmacro check% (form)
  `(report-result ,form ',form))

;; so? we get this one
(defun test-+%%%% ()
  (check% (= (+ 1 2) 3))
  (check% (= (+ 1 2 3) 6))
  (check% (= (+ 1 -3) -2)))

;; still repetive, go further
(defmacro check (&body forms)
  `(progn
     ,@(loop for f in forms collect `(report-result ,f ',f))))

;; so very clean!
(defun test-+ ()
  (check
    (= (+ 1 2) 3)
    (= (+ 1 2 3) 6)
    (= (+ 1 -4) -2)))
  
;; Problem 5: can the return value of test-+ indicate whether success or not?
;; STEP 5: let the print function return the result

;; Override!
(defun report-result (result form)
  (format t "~:[FAIL~;pass~] ... ~a~%" result form)
  result)

;; to result all result of tests, we can not use a ADD for its shortcut
;; we need something act as follow because we need both the result and
;; the full execution.
;;
;; (let ((result t))
;;   (unless (foo) (setf result nil))
;;   (unless (bar) (setf result nil))
;;   (unless (baz) (setf result nil))
;;   result)
;;
(defmacro combine-results (&body forms)
  (let ((result (gensym)))
    `(let ((,result t))
       ,@(loop for f in forms collect `(unless ,f (setf ,result nil)))
       ,result)))

;; so we get a new check
(defmacro check (&body forms)
  `(combine-results
     ,@(loop for f in forms collect `(report-result ,f ',f))))

;; recompile test-+ we can get the return value


;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; OK. What happen if we want to test more than one function?

;; for example
(defun test-* ()
  (check
   (= (* 2 2) 4)
   (= (* 3 5) 15)))

;; It's easy, right? Just a function!
(defun test-arithmetic ()
  (test-*)
  (test-+))

;; when there are lots of tests, it will be very difficult to figure out
;; which test in which function fail
;; if the test can report its function name, it will be helpful

;; Problem 6: update report function to provide the function name of test
;; STEP 6: we need a dynamic variable
(defvar *test-name* nil)

;; update the print function
(defun report-result (result form)
  (format t "~:[FAIL~;pass~] ... ~a: ~a~%" result *test-name* form)
  result)

;; test function should modify *test-name*
(defun test-+ ()
  (let ((*test-name* 'test-+))
    (check
      (= (+ 1 2) 3)
      (= (+ 1 2 3) 6)
      (= (+ 1 -4) -2))))

(defun test-* ()
  (let ((*test-name* 'test-*))
    (check
      (= (* 2 2) 4)
      (= (* 3 5) 15))))
;; run the test-arithmetic we get this 
;; pass ... TEST-*: (= (* 2 2) 4)
;; pass ... TEST-*: (= (* 3 5) 15)
;; pass ... TEST-+: (= (+ 1 2) 3)
;; pass ... TEST-+: (= (+ 1 2 3) 6)
;; FAIL ... TEST-+: (= (+ 1 -4) -2)    

;; it seem good. But it is painful to write a lot of boilplate code!
;; We have to modify every test-function!
;; All the repetition is bad!
;; where there is a repetition, there an abstraction is needed!

;; what we try to catch is a DEFUN and some boilplate, let's write a macro!
(defmacro deftest (name parameters &body body)
  `(defun ,name ,parameters
     (let ((*test-name* ',name))
       ,@body)))

;; using deftest to rewrite test function --- abstraction over function!
(deftest test-+ ()
  (check
      (= (+ 1 2) 3)
      (= (+ 1 2 3) 6)
      (= (+ 1 -4) -2)))

(deftest test-* ()
  (check
    (= (* 2 2) 4)
    (= (* 3 5) 15)))

;; can we do better?
;; when our test function is a part of some big project, it is far from enough
;; to know which test fail but which function contains the test!
;; Problem 7: we need a test layer
;; STEP 7: save every function name in the dynamic variable

;; override!
(defmacro deftest (name parameters &body body)
  `(defun ,name ,parameters
     (let ((*test-name* (append *test-name* (list ',name))))
       ,@body)))

;; using the Deftest to override the test-arithmetic
(deftest test-arithmetic ()
  (combine-results
    (test-*)
    (test-+)))

;; now we get:

;; CL-USER> (test-arithmetic)
;; pass ... (TEST-ARITHMETIC TEST-*): (= (* 2 2) 4)
;; pass ... (TEST-ARITHMETIC TEST-*): (= (* 3 5) 15)
;; pass ... (TEST-ARITHMETIC TEST-+): (= (+ 1 2) 3)
;; pass ... (TEST-ARITHMETIC TEST-+): (= (+ 1 2 3) 6)
;; FAIL ... (TEST-ARITHMETIC TEST-+): (= (+ 1 -4) -2)

;; Perfect!
