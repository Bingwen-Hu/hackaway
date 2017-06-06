;;; chapter 4.1

;; change the binding of stdout to stderr
(binding [*out* *err*]
  (println "I am in error!"))

;; print to a file
(def ifile (clojure.java.io/writer "ifile.txt"))
(binding [*out* ifile]
  (println "Mory file."))

(.close ifile)
