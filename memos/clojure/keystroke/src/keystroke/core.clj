(ns keystroke.core
  (:import [jline.console ConsoleReader]))

(defn show-keystroke []
  (print "Enter a keystroke: ")
  (flush)
  (let [cr (ConsoleReader.)
        keyint (.readCharacter cr)]
    (println (format "Got %d ('%c')!" keyint (char keyint)))))
