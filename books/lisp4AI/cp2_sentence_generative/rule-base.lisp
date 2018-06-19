(defparameter *simple-grammar*
  '((sentence -> (noun-phrase verb-phrase))
    (noun-phrase -> (Article Noun))
    (verb-phrase -> (Verb noun-phrase))
    (Article -> the a)
    (Noun -> man ball woman table)
    (Verb -> hit took saw liked))
  "A grammar for a trivial subset of English.")

(defvar *grammar* *simple-grammar*
  "The grammar used by generate. Initially, this is *simple-grammar*,
but we can switch to other grammars.")


(defun rule-lhs (rule)
  "the left-hand side of a rule."
  (first rule))

(defun rule-rhs (rule)
  "the right-hand side of a rule."
  (rest (rest rule)))

(defun rewrites (category)
  "return a list of the possible rewrites for this category."
  (rule-rhs (assoc category *grammar*)))


(defun mappend (fn lst)
  (reduce #'append (mapcar fn lst)))

(defun random-elt (lst)
  (elt lst (random (length lst))))

(defun generate (phrase)
  "Generate a random sentence or phrase"
  (cond 
    ((listp phrase)
     (mappend #'generate phrase))
    ((rewrites phrase)
     (generate (random-elt (rewrites phrase))))
    (t (list phrase))))
	

(defparameter *Bigger-grammar*
  `((sentence -> (noun-phrase verb-phrase))
    (noun-phrase -> (Article Adj* Noun PP*) (Name) (Pronoun))
    (verb-phrase -> (Verb noun-phrase PP*))
    (PP* -> () (PP PP*))
    (Adj* -> () (Adj Adj*))
    (PP -> (Prep noun-phrase))
    (Prep -> to in by with on)
    (Adj -> big little blue green adiabatic)
    (Article -> the a)
    (Name -> Pat Kim Lee Terry Robin)
    (Noun -> man ball woman table)
    (Verb -> hit took saw liked)
    (Pronoun -> he she it these those that)))

(setf *grammar* *bigger-grammar*)


;; exercise
(defun cross-product (fn xlist ylist)
  (mappend #'(lambda (y)
	       (mapcar #'(lambda (x) (funcall fn x y))
		       xlist))
	   ylist))