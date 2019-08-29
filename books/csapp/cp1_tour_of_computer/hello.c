#include <stdio.h>

int main(){
    printf("Hello, world\n");
}


/* source program is a sequence of bits, each with a value of 0 or 1. 
 * organzied in 8-bit chunks called bytes.
 *
 * gcc hello.c -o hello
 * consist of four steps:
 * preprocessor -> compiler -> assembler -> linker
 * known as compilation system
 *
 * Preprocessing phase: expand macro (#) generates hello.i
 * Compilation phase: cc1 translates the text file hello.i to assemble 
 *                    language file hello.s
 * Assembly phase: assembler(as) translates hello.s into machine-language
 *                 instructions, packages them in a form called relocatable
 *                 object program hello.o
 * Linking phase: merge other precompiled object files into an executable
 *                object file hello.
 * */


