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


;;; from joy of clojure
(defn join
  {:test (fn []
           (assert
            (= (join "," [1 2 3]) "1,2,3")))}
  [sep s]
  (apply str (interpose sep s)))



;;; high order function
(def plays [{:band "Burial", :plays 979, :loved 9}
            {:band "Eno", :plays 2333, :loved 15}
            {:band "Bill Evans", :plays 979, :loved 9}
            {:band "Magma", :plays 265, :loved 31}])

(def sort-by-loved-ratio 
  (partial sort-by #(/ (:plays %) (:loved %))))

(defn columns [column-names]
  (fn [row]
    (vec (map row column-names))))

(sort-by (columns [:plays :loved :band]) plays)


;;; named parameter
(defn slope
  [& {:keys [p1 p2] 
      :or [p1 [0 0]
           p2 [0 0]]}]
  (float (/ (- (p2 1) (p1 1))
            (- (p2 0) (p1 0)))))

;;; assert
(defn slop [p1 p2]
  {:pre [(not= p1 p2) (vector? p1) (vector p2)]
   :post [(float? %)]}
  (/ (- (p2 1) (p1 1))
     (- (p2 0) (p1 0))))

;;; assert from outside
(defn put-things [m]
  (into m {:meat "beef", :veggie "broccoli"}))

(defn balanced-dist [f m]
  {:post [(:meat %) (:veggie %)]} ;; must have meat and veggie
  (f m))


(defn finicky [f m]
  {:post [(=  (:meat %) (:meat "beef"))]} ;; must beef
  (f m))
