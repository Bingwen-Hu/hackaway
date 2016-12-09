### the room command inspect the memory already used.

    CL-USER> (room)
    Dynamic space usage is:   58,576,288 bytes.
    Read-only space usage is:      4,912 bytes.
    Static space usage is:         3,216 bytes.
    Control stack usage is:        8,552 bytes.
    Binding stack usage is:        1,088 bytes.
    Control and binding stack usage is for the current thread only.
    Garbage collection is currently enabled.

    Breakdown for dynamic space:
      15,138,928 bytes for    20,297 code objects.
      11,485,872 bytes for   131,397 simple-vector objects.
      11,007,616 bytes for   687,976 cons objects.
      9,912,512 bytes for   127,333 instance objects.
      13,085,712 bytes for   214,459 other objects.
      60,630,640 bytes for 1,181,462 dynamic objects (space total.)
      
### describe command inspect the element 

    CL-USER> (describe north-maple)
    #S(SURPASSER :NAME NORTH-MAPLE :PROPERTY WIND :LEADER ANDIAN)
      [structure-object]

      Slots with :INSTANCE allocation:
        NAME      = NORTH-MAPLE
	PROPERTY  = WIND
	LEADER    = ANDIAN

### the Inspect command (entry a new REPL for inspect)

    CL-USER> (inspect north-maple)

    The object is a STRUCTURE-OBJECT of type SURPASSER.
     0. NAME: NORTH-MAPLE
     1. PROPERTY: WIND
     2. LEADER: ANDIAN

### the time command tell the time the procedure take

    CL-USER> (defun addup (n)
	     	(do ((i 0 (1+ i))
		     (sum 0 (+ sum i)))
	       	    ((> i n) sum)))
    CL-USER> (time (addup 1000000))
    Evaluation took:
      0.023 seconds of real time
      0.020000 seconds of total run time (0.020000 user, 0.000000 system)
      86.96% CPU
      49,305,400 processor cycles
      0 bytes consed
  
    500000500000

