(define atom?
   (lambda (x)
    (and (not (pair? x))
         (not (null? x)))))

;; In these section, we learn
;; atom? eq? car cdr null?


(define lat?        
  (lambda (lst)
    (cond 
     ((null? lst) #t)
     ((atom? lst) #f)                   ;false or error?
     ((atom? (car lst)) (lat? (cdr lst)))
     (else #f))))
