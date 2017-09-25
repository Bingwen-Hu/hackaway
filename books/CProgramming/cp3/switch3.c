#include <stdio.h>


int main(){

  int i = 0;

  switch(i = i + 4){

    case 4:
      printf("You have passed");
      break;
    default:
      printf("Failed!");
      break;
  }

  return 0;
}


/* Test Note:
   
   i = i + 4 return the value of i(left value) so the switch 
   statement succeeds.
 */
