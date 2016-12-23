;;;; Recursively Specified Data

;; Inductive Specification
;; how to define the recursively struction or expression

;; definition 1 -- top-down 
;; 1. n = 0 or
;; 2. n - 3 in S

(defun in-S (n)
  (cond ((zerop n) t)
	((>= (- n 3) 0)
	 (in-S (- n 3)))))
    
;; definition 2 -- bottom-up
;; 0 in S and
;; if n in S, then n + 3 in S
;; another expression call rule-of-inference is a shortcut of definition 2



;; defining sets using grammars
;; List-of-Int ::= ()
;; List-of-Int ::= (Int . List-of-Int)

;; three concepts about the grammars
;; Nonterminal Symbols: the name of the sets being defined (left side)
;; Terminal Symbols: right side. contains some .;() maybe nonterminal symbols
;; Productions: the rule as a whole are called productions.

;; shortcut
;; List-of-Int ::= ()
;;             ::= (Int . List-of-Int
;; List-of-Int ::= () | (Int . List-of-Int)
;; List-of-Int ::= ({Int}*)

;; s-list and s-expression
;; S-list ::= ({S-exp}*)
;; S-exp ::= Symbol | S-list

;; lambda expression
;; LcExp ::= Identifier
;;       ::= (lambda (Identifier) LcExp)
;;       ::= (LcExp, LcExp)
;; where an identifier is any symbol other than lambda

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Proof by Structural Induction
;; To prove that a proposition IH(s) is true for all structure s,
;; prove the following:
;; 1. IH is true on simple structures (those without substructures)
;; 2. If IH is true on the substructures of s, then it is true on s
;;    itself.
;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
