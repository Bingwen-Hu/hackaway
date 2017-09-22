#include <stdio.h>


int main(){
  
  int i = 2;
  printf("%d %d\n", ++i, ++i);


  int x = 10, y = 20, z = 5, j;

  j = x < y < z;

  printf("%d\n", j);
  

  int a = 10, b, c;
  b = c = 0;

  a<=20 ? b = 30 : (c = 30);

  printf("b=%d, c=%d\n", b, c);

  
  return 0;

}


/* Test Note:
   
   printf("%d %d\n", ++i, ++i);
   1. increase i, i => 3
   2. increase i, i => 4
   3. printf("%d %d\n", 4, 4)
   

   j = x < y < z 
   1. eval y < z and discard the result
   2. eval x < y and assign to j
   
 */


