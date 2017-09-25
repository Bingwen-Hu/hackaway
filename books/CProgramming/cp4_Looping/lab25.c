#include <stdio.h>

int main(){

  int n = 5;

  for(int i=1; i<=n; i++) {
    
    for(int j=n-i; j>=1; j--)
      putchar(' ');
    
    for(int k=1; k<=i*2-1; k++)
      printf("%d", k<=i ? k : (2*i-k));

    for(int l=n-i; l>=1; l--)
      putchar(' ');
    
    putchar('\n');
  }


  return 0;
}


/* Test Note
   
   The hardest way is easies way to learn C.

   It's worth to dive into a small program that finally get a 
   beautiful result.
   
   Everything happens for some reason.

 */
