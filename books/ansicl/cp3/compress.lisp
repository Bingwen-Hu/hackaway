(defun compress (x)
  (if (consp x)
      (mory-compr (car x) 1 (cdr x))
      x))

(defun compr (elt n lst)
  (if (null lst)
      (list (n-elts elt n))
      (let ((next (car lst)))
        (if (eql next elt)
            (compr elt (+ n 1) (cdr lst))
            (cons (n-elts elt n)
                  (compr next 1 (cdr lst)))))))

(defun mory-compr (elt n lst)
  "mory version"
  (labels ((compr-iter (elt n lst ret)
             (if (null lst)
                 (cons (n-elts elt n) ret)
                 (let ((next (car lst)))
                   (if (eql next elt)
                       (compr-iter elt (+ n 1) (cdr lst) ret)
                       (compr-iter next 1 (cdr lst) (cons
                                                     (n-elts elt n)
                                                     ret)))))))
    (nreverse (compr-iter elt n lst nil))))
(defun n-elts (elt n)
  (if (> n 1)
      (list n elt)
      elt))

(defun uncompress (lst)
  (if (null lst)
      nil
      (let ((elt (car lst))
            (rest (uncompress (cdr lst))))
        (if (consp elt)
            (append (apply #'list-of elt)
                    rest)
            (cons elt rest)))))

(defun list-of (n elt)
  (if (zerop n)
      nil
      (cons elt (list-of (- n 1) elt))))


(list-of 3 'ho)
(uncompress '((3 1) 0 1 (4 0) 1))
(compress '(3 3 3 1 1 1 0 1 2))


