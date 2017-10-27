(with-input-from-string (stream "This is my input stream!")
  (read stream)
  (read stream)
  (read stream))

(with-output-to-string (stream)
  (princ "Mory and Ann will never leave apart!" stream))

