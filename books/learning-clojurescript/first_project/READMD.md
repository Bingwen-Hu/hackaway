# how this work?
1. download clojurescript compiler from its github main page
2. export path of the compiler as CLJS
3. setup first_project.core and build.clj ready
4. run `java -cp $CLJS:src clojure.main build.clj`