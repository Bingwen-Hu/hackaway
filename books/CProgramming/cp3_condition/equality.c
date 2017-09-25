#include <stdio.h>

int main(){

  int a = 3;
  float b = 3.0;

  if (a == b) 
    printf("floats equal to integers");
  else
    printf("floats inequal to integers.");


  int k = 30;

  printf("%d %d %d %d: ", k, k==30, k=50, k>40);


  return 0;
}


/* Test Note:

   1. floats and integers can be equal.

   2. in printf statement, parameter expressions are evaluated from right 
   to left, so k in finally become 50 and k==35 is false!

 */
