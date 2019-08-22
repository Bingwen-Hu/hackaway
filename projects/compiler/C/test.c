// this is a test c program to test our own C compile
int main() {
    return 42;
}


// Here are some notes
// We won’t transform the assembly into an executable ourselves - that’s the job 
// of the assembler and linker, which are separate programs

/** code
 * $ ./YOUR_COMPILER return_2.c # compile the source file shown above
 * $ ./gcc -m32 return_2.s -o return_2 # assemble it into an executable
 * $ ./return_2 # run the executable you just compiled
 * $ echo $? # check the return code; it should be 2
 * 2 
 */

