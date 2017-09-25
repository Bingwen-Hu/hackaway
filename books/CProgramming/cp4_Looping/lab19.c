#include <stdio.h>

int main(){
  
  int n = 5;
  
  for (int i=1; i<=5; i++) {
    
    for (int j=1; j<=i; j++)
      putchar('*');
    
    putchar('\n');
  }

  return 0;
}


/* Test Note
   
   lab20 is same as lab19

 */
