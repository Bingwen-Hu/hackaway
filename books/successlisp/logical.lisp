;; AND and OR
(defun safe-elt (sequence index)
  (and (< -1 index (length sequence))
       (values (elt sequence index) t)))



;; boole function
(boole boole-and 15 7)
(boole boole-ior 2 3)
(boole boole-set 99 55)

;; bitwise logical function
(logtest 7 16)
(logtest 15 5)
(logbitp 0 16)
(logbitp 4 16)
(logcount 35)

;; bit vector
(length #*0010101)
(bit-and #*00110100 #*10101010)
