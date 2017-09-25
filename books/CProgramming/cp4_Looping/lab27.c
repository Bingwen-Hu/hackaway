#include <stdio.h>

int main(){
  
  int n = 4;

  for(int i=1; i<=n*2-1; i++) {
    for(int j=n-i; j>=1; j--)
      putchar(' ');

    for(int k=1; k<=i*2-1; k++)
      putchar('*');
    

    putchar('\n');
    
  }

  
  return 0;
}


/* Test note
   
   fail to teckle

 */
