#include <stdio.h>

int main(){

  int a[10] = {1};

  
  for (int i=0; i<20; i++)
    printf("%ld ", a[i]);
  
  putchar('\n');
}


/* Test Note:
   
   boundary is never check in C
   Rust can do it better!
 */
