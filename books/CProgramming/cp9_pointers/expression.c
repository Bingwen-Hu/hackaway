#include <stdio.h>

void main(){

  int lst[5] = {10, 20, 30, 40, 50}, *p;

  p = lst;

  printf("%d, ", *++p);
  printf("%d, ", *p--);
  printf("%d, ", *p);

  printf("%d", *(p+2));


  
}


/* Test note:

   pointer plus some integer will get a valid number.
   it moves ......

 */
