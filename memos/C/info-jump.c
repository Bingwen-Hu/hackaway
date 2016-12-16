/* Modified version from tlpi by Michael Kerrisk - Page 134 */

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>

static jmp_buf env;

static void f2()
{
    longjmp(env, 2);
}

static void f1(int argc)
{
    if (argc == 1)
	longjmp(env, 1);
    f2();
}

int main(int argc, char *argv[])
{
    switch(setjmp(env)) {
    case 0:
	printf("Calling f1() after initial setjmp() \n");
	f1(argc);		/* never returns .. */
	break;			/* ... but this is good form */

    case 1:
	printf("we jumped back from f1() \n");
	break;
	
    case 2:
	printf("we jumped back from f2() \n");
	break;
    }

    exit(EXIT_SUCCESS);
}
/* * *

Compile and running: 
demon@linux:~> gcc info-jump.c 
demon@linux:~> ./a.out 
Calling f1() after initial setjmp() 
we jumped back from f1() 
demon@linux:~> ./a.out 1
Calling f1() after initial setjmp() 
we jumped back from f2() 


The procedure for first call
function-main::setjmp(env) -> return 0 because it is initial ->
switch(0) -> case 0 -> printf(...) -> call function-f1(argc) ->
f1::if -> if::longjmp -> return to function-main::setjmp -> 
switch(1) -> case 1 -> printf(...) -> break -> exit 

the procdure for second call is similar but jump to case 2 rather 
than case 1

   


* * */
