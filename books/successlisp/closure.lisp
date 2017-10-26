(let ((password nil)
      (secret nil))
  (defun set-password (new-passwd)
    (if password
	'|Can't -already set|
	(setq password new-passwd)))
  (defun change-password (old-passwd new-passwd)
    (if (eq old-passwd password)
	(setq password new-passwd)
	'|Not changed|))
  (defun set-secret (passwd new-secret)
    (if (eq passwd password)
	(setq secret new-secret)
	'|Wrong password|))
  (defun get-secret (passwd)
    (if (eq passwd password)
	secret 
	'|Sorry|)))