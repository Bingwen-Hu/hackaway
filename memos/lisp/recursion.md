# Recursion Templates

### double-test tail recursion

    Template:
    (defun func (x)
        (cond (end-test-1 end-value-1)
	      (end-test-2 end-value-2)
	      (T (func reduced-x))))

    Example:
    (defun anyoddp (x)
        (cond ((null x) nil)
	      ((oddp (first x)) t)
	      (t (anyoddp (rest x)))))


### single-test tail recursion

    Template:
    (defun func (x)
        (cond (end-test end-value)
	      (T (func reduced-x))))

    Example:
    (defun find-first-atom (x)
        (cond ((atom x) x)
	      (t (find-first-atom (first x)))))

### single-test augmenting recursion

    Template:
    (defun func (x)
        (cond (end-test end-value)
	      (T (aug-fun aug-val
	                  (func reduced-x)))))

    Example:
    (defun count-slices (x)
        (cond ((null x) 0)
	      (t (+ 1 (count-slices (rest x))))))

### list-consing recursion

    Template:
    (defun func (n)
        (cond (end-test nil)
	      (T (cons new-element
	               (func reduced-x)))))

    Example:
    (defun laugh (n)
        (cond ((zerop n) nil)
	      (t (cons 'ha (laugh (- n 1))))))

### simultaneous recursion on several variables

    Template:
    (defun func (n x)
        (cond (end-test end-value)
	      (T (func reduced-n reduced-x))))

    (defun my-nth (n x)
        (cond ((zerop n) (first x))
	      (t (my-nth (- n 1) (rest x)))))

### conditional augmentation

    Template:
    (defun func (x)
        (cond (end-test end-value)
	      (aug-test (aug-fun aug-val
	                         (func reduced-x))
	      (T (func reduced-x)))))

    Example:
    (defun extract-symbols (x)
        (cond ((null x) nil)
	      ((symbolp (first x))
	       (cons (first x)
	             (extract-symbols (rest x))))
	      (t (extract-symbols (rest x)))))

### multiple recursion

    Template:
    (defun func (n)
        (cond (end-test-1 end-value-1)
	      (end-test-2 end-value-2)
	      (t (combiner (func first-reduced-n)
	      	 	   (func second-reduced-n)))))

    Example:
    (defun fib (n)
        (cond ((equal n 0) 1)
	      ((equal n 1) 1)
	      (t (+ (fib (- n 1))
	      	    (fib (- n 2))))))

### car/cdr recursion

    Template:
    (defun func (x)
        (cond (end-test-1 end-value-1)
	      (end-test-2 end-value-2)
	      (t (combiner (func (car x))
	      	 	   (func (cdr x))))))

    Example:
    (defun find-number (x)
        (cond ((numberp x) x)
	      ((atom x) nil)
	      (t (or (find-number (car x))
	      	     (find-number (cdr x))))))
