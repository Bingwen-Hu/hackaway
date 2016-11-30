# operation system

### run C programs
	
	- hello.c(text) => Preprocessor(cpp) => hello.i(text)
	- hello.i(text) => Compiler(ccl) => hello.s(text)
	- hello.s(text) => Assembler(as) => hello.o(binary)
	- hello.o(binary) + lib* => linker(ld) => hello(binary, ready to run)

### system components

	- CPU => program counter(PC), ALU, Register file
	- system bus & I/O bridge
	- memory bus & main memory
	- I/O bus & USB controller, graphics adapter, disk controller, etc 

### abstractions
	
	- files => I/O devices
	- virtual memory => files + main memory
	- processes => virtual memory + processor
	- virtual machine => processes + operationg system

### Amdahl's law

	when we speed up one part of a system, the effect on the overall system performance depends on both how significant this part was and how much it sped up.
	
### byte's order
	
	assume the int value is 0x12345678, the address is 0100 0101 0102 0103
	- big endian 
	  12 34 56 78
	- small endian
	  78 56 34 12
	
### numbers, boolean and bitwise

	- boolean operation yield 1 or 0, using !, && and ||
	- bitwise operation yield numbers, using ~(not), ^(not or), &(and), |(or)
	- left shift: a << 3 always means a * 2^3, append 0s on the least side
	- right shift: a >> 3 always means a / 2^3, append 0s at the highest side logically and 1s arithmeticly
	- multi and division is costly, always changed to add, shift, and sub.	

### signed and unsigned

	- 1111(u) means 2^3 + 2^2 + 2^1 + 2^0 = 15
	- 1111(t) means -2^3 + 2^2 + 2^1 + 2^0 = -1
	- 1000(u) means 8 while 1000(s) means -8

	overflow
	- 1000(u) + 1000(u) = [10000] => [0000] so 8 + 8 = 0
	- 1000(s) + 1000(s) = [10000] => [0000] so -8 + -8 = 0
	
	two's complement
	- 1100 == 11100 == 111100 ...... signed matters!
