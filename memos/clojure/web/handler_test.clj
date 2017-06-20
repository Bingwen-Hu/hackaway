(ns web.handler-test
  (:require [clojure.test :refer :all]
            [ring.mock.request :as mock]
            [web.handler :refer :all]))

(deftest test-app
  (testing "main route"
    (let [response (app (mock/request :get "/"))]
      (is (= (:status response) 200))
      (is (= (:body response) "Hello World"))))

  (testing "echo route"
    (let [response (app (mock/request :post "/echo" "Echo!"))]
      (is (= 200 (:status response)))
      (is (= "Echo!" (:body response))))
    (let [response (app (mock/request :post "/echo"))]
      (is (= 400 (:status response)))))

  (testing "not-found route"
    (let [response (app (mock/request :get "/invalid"))]
      (is (= (:status response) 404)))))

(deftest catchall-test
  (testing "when a handler throws an exception"
    (let [response (app (mock/request :get "/trouble"))]
      (testing "the status code is 500"
        (is (= 500 (:status response))))
      (testing "and the body only contains the exception message"
        (is (= "Divide by zero" (:body response)))))))

(deftest slurp-body-test
  (testing "when a handler requires a request body"
    (testing "and a body is provided"
      (let [response (body-echo-app (mock/request :post "/" "Echo!"))]
        (testing "the status code is 200"
          (is (= 200 (:status response)))
          (testing "with the request body in the response body"
            (is (= "Echo!" (:body response)))))))
    (testing "and a body is not provided"
      (let [response (body-echo-app (mock/request :get "/"))]
        (testing "the status code is 400"
          (is (= 400 (:status response))))))))


(deftest links-test
  (testing "the links/:id endpoint"
    (testing "when an id is provided"
      (let [response (app (mock/request :get "/links/foo13"))]
        (testing "returns a 200"
          (is (= 200 (:status response)))
          (testing "with the id in the body"
            (is (re-find #"foo13" (:body response)))))))
    (testing "when the id is omitted"
      (let [response (app (mock/request :get "/links"))]
        (testing "returns a 404"
          (is (= 404 (:status response))))))
    (testing "when the path is too long"
      (let [response (app (mock/request :get "/links/are/you/kidding"))]
        (testing "returns a 404"
          (is (= 404 (:status response))))))))
