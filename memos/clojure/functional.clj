;;; clojure functional tools

;;; map reduce
(map #(inc %) [1 2 3])
(reduce + [1 2 3])

;;; apply partial
(def args [1 -2 2])
(apply * 4 args)

((partial map *) [1 2 3] [3 4 5])


;;; comp is very powerful
(defn negated-sum-str
  [& numbers]
  (str (- (apply + numbers))))

(defn negated-sum-str 
  [comp str - +])

;;; alias just meet
(require '[clojure.string :as str])
;;; little tool and note that #"" means regular expression
(def camel->keyword (comp keyword
                          str/join
                          (partial interpose \-)
                          (partial map str/lower-case)
                          #(str/split % #"(?<=[a-z])(?=[A-Z])")))
