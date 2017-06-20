(ns web.handler
  (:require [compojure.core :refer :all]
            [compojure.route :as route]
            [ring.middleware.defaults :refer [wrap-defaults api-defaults]]))


(require '[ring.util.response :as ring-response])
(defn wrap-500-catchall
  [handler]
  (fn [request]
    (try (handler request)
         (catch Exception e
           (-> (ring-response/response (.getMessage e))
               (ring-response/status 500)
               (ring-response/content-type "text/plain")
               (ring-response/charset "utf-8")
               )))))


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

(defn echo 
  [body]
  (if (not-empty body)
    (-> (ring-response/response body)
        (ring-response/content-type "text/plain")
        (ring-response/charset "utf-8"))
    (-> (ring-response/response "You must submit a body with your request!")
        (ring-response/status 400))))



;;; app-routes is a handler!
(defroutes non-body-routes
  (GET "/" [] "Hello World")
  (GET "/trouble" [] (/ 1 0))           ; this won't end well!
  (GET "/links/:id" [id] (str "The id is: " id))
  (route/not-found "Not Found"))

;;; a handler
(def body-routes 
  (-> (routes 
       (ANY "/echo" [:as {body :body}] (echo body)))
      (wrap-routes wrap-slurp-body)))


(def app-routes 
  (routes body-routes non-body-routes))

;;; a handler
(def app
  (-> app-routes
      wrap-500-catchall
      (wrap-defaults api-defaults)))

