#include <stdio.h>

int main(){

  int i = 3;

  switch(i){

    default:
      printf("Matrix");
      break;
     case 1:
       printf("Computer");
       break;
     case 2:
       printf("Education");
       break;
     case 4:
       printf("Hello");
       break;
  }

  return 0;
}


/* Test Note:
   
   the Switch will look up every case statement and make a decision,
   although the default case is the first, it makes no difference.

 */
