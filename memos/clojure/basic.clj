;;; Not that unchangable state is a norm in Clojure

;;; very basic
(defn average
  [numbers]
  (/ (apply + numbers) (count numbers)))

;;; in fact, defn is a macro
(def average
  (fn average 
    [numbers] 
    (/ (apply + numbers) (count numbers))))

;;; eval is the extreme tool but used infrequently, macro can do.
(eval :foo)
(eval [1 2 3])
(eval "text")
(eval '(average [60 80 100 400]))
(eval (read-string "(average [60 100 80 200])"))
(defn embedded-repl 
  "A naive Clojure REPL implementation. Enter ':quit' to exit."
  []
  (print (str (ns-name "ns") ">>> "))
  (flush)
  (let [expr (read)
        value (eval expr)]
    (when (not= :quit value)
      (println value)
      (recur))))

;;; all the interoperation with Java depend on new and . operation

;;; define a var and get its reference
(def x 5)
x                                       ; value
(var x)                                 ; reference or 
#'x                                     ; same as (var x)

;;; loop
(loop [x 5]
  (if (neg? x)
    x
    (recur (dec x))))                   ; recur pass the control to loop and bind the new x

(defn countdown
  [x]
  (if (zero? x)
    :blastoff!
    (do (println x)
        (recur (dec x)))))              ; recur pass the control to function as bind the new parameter

;;; note that recur is a low-level tool, upon which there are tools like doseq and dotimes


;;; function literal
;;; lambda function
(fn [x y] (Math/pow x y))

;;; same as
#(Math/pow %1 %2)

;;; Note that function literal doesn't have a do but fn has
(fn [x y]
  (println (str x \^ y))
  (Math/pow x y))

#(do (println (str %1 \^ %2)              ; you need a do
              (Math/pow %1 %2)))

(fn [x]                                   ; fn can have fn 
  (fn [y]                                 ; but # can't
    (+ x y)))



