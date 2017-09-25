#include <stdio.h>

int main(){

  int i = 1;
  
  switch(i) {
    
    printf("This statement will never execute\n");

    case 1:
      printf("Mory\n");
      break;
    case 2:
      printf("Ann\n");
      break;
    default:
      printf("Surpasser\n");
      break;
  }
  return 01;
}


/* Test note:
   
   return value 01 is valid, invalid in Python!
   
   in switch statement, every command should be in the case statement to 
   get executed.


   Another Note:
   switch can work only on integer constanst and constant expressions
 */
