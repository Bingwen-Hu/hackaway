(ns web.handler
  (:require [compojure.core :refer :all]
            [ring.util.response :as ring-response]
            [ring.middleware.json :as ring-json]
            [compojure.route :as route]
            [cheshire.core :as json]
            [ring.middleware.defaults :refer [wrap-defaults 
                                              api-defaults
                                              site-defaults]]))

;;; middleware
(defn wrap-500-catchall
  [handler]
  (fn [request]
    (try (handler request)
         (catch Exception e
           (-> (ring-response/response (.getMessage e))
               (ring-response/status 500)
               (ring-response/content-type "text/plain")
               (ring-response/charset "utf-8"))))))

;;; middleware
(defn wrap-slurp-body
  [handler]
  (fn [request]
    (if (instance? java.io.InputStream (:body request))
      (let [prepared-request (update request :body slurp)]
        (handler prepared-request))
      (handler request))))

;;; test wrap-slurp-body
(defn body-echo-handler
  [request]
  (if-let [body (:body request)]
    (-> (ring-response/response body)
        (ring-response/content-type "text/plain")
        (ring-response/charset "utf-8"))
    (-> (ring-response/response "You must submit a body with your request!")
        (ring-response/status 400))))

(def body-echo-app
  (-> body-echo-handler
      wrap-500-catchall
      wrap-slurp-body))

;;; a handler-like function
(defn echo 
  [body]
  (if (not-empty body)
    (-> (ring-response/response body)
        (ring-response/content-type "text/plain")
        (ring-response/charset "utf-8"))
    (-> (ring-response/response "You must submit a body with your request!")
        (ring-response/status 400))))

;;; json middleware
(defn wrap-json
  "turn the string to json"
  [handler]
  (fn [request]
    (if-let [prepd-request (try (update request :body json/decode)
                                (catch com.fasterxml.jackson.core.JsonParseException e
                                  nil))]
      (handler prepd-request)
      (-> (ring-response/response "Sorry, that's not Json.")
          (ring-response/status 400)))))

;;; middleware
(defn handle-clojurefy
  "turn the json to clojure map"
  [request]
  (-> (:body request)
      str
      ring-response/response
      (ring-response/content-type "application/edn")))


;;; json
(def handle-info
  (ring-json/wrap-json-response
   (fn [_] 
     (-> {"Java Version" (System/getProperty "java.version")
          "OS Name" (System/getProperty "os.name")
          "OS Version" (System/getProperty "os.version")}
         ring-response/response))))

;;; app-routes is a handler!
(defroutes non-body-routes
  (GET "/" [] "Hello World")
  (GET "/trouble" [] (/ 1 0))           ; this won't end well!
  (GET "/links/:id" [id] (str "The id is: " id))
  (GET "/info" [] handle-info)
  (route/not-found "Not Found"))

(def json-routes
  (routes
   (POST "/clojurefy" [] (ring-json/wrap-json-body handle-clojurefy))))

(def body-routes 
  (-> (routes 
       (ANY "/echo" [:as {body :body}] (echo body)))
      (wrap-routes wrap-slurp-body)))

;;; combine two routes
;;; Note that the order matters, the non-body-routes should be place the last for it contains 
;;; the not-found route!
(def app-routes
  (routes body-routes json-routes non-body-routes))

;;; a final route
(def app
  (-> app-routes
      wrap-500-catchall
      (wrap-defaults api-defaults)))

