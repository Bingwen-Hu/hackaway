; proper list: one whose last cdr is NIL
; dotted list: last cons is a dotted pair

;; copy-list doesn't copy the sublists, like shadow-copy
;; copy-tree copy all the cons, like deep-copy

;; tree base replace: subst
(subst 10 1 '(1 2 (3 2 1) ((1 1) (2 2))))

; subst-if-not nsubst nsubst-if nsubst-if-not

;;; list as set
(defparameter *set* ())
(adjoin 1 *set*)
(setf *set* (adjoin 1 *set*))
(pushnew 2 *set*)
(pushnew 2 *set*)

(member 1 *set*)
(set-difference *set* '(3 2))
(set-exclusive-or *set* '(3 2))
(intersection *set* '(3 2))
(subsetp *set* *set*)


;; associate list or proper list
(assoc 'a '((a . 1) (b . 2) (c . 3)))
(assoc 'd '((a . 1) (b . 2) (c . 3)))

(assoc "a" '(("a" . 1) ("b" . 2) ("c" . 3))) ;=> nil
(assoc "a" '(("a" . 1) ("b" . 2) ("c" . 3)) :test #'string=)

(cdr (assoc 'a '((a . 1) (b . 2) (c . 3))))
;; shadow
(assoc 'a '((a . 1) (a . 10) (b . 2) (c . 3)))

;; modify alists
(defparameter alist nil)
(setf alist (acons 'new-key 'new-value  alist))
(push (cons 'new-key2 'new-value2) alist)

(pairlis '(k1 k2 k3) '(v1 v2 v3))

;; plists
(defparameter *plist* ())
(setf (getf *plist* :a) 1)
(remf *plist* :a)

;; plist in symbol
(setf (get 'mory 'age) 20)
(setf (get 'mory 'surpass) 'heart)


;; destructuring-bind
(destructuring-bind (x y z) (list 1 2 3)
  (list :x x :y y :z z))

(destructuring-bind (x (y1 y2) z) '(1 (2 2) 3)
  (list :x x :y1 y1 :y2 y2 :z z))

(destructuring-bind (&key x y z) '(:y 2 :z 3 :x 1)
  (list :x x :y y :z z))

; whole
(destructuring-bind (&whole whole &key x y z) '(:y 2 :z 3 :x 1)
  (list :x x :y y :z z :whole whole))
