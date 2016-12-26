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

