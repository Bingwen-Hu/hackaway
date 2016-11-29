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
	

