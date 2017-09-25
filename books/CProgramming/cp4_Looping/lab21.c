#include <stdio.h>


int main(){

  int n = 9;

  for (int i=1; i<=n; i+=2) {
    for (int j=1; j<=i; j++)
      printf("%d", j);
    
    putchar('\n');
  }
  
  return 0;
}
