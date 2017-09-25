#include <stdio.h>

int main(){

  int n = 5;                    /* pretend user input */

  for (int i=1; i<=5; i++) {
    
    for (int j=i; j>0; j--) {
      printf("%d", j);
    }
    
    putchar('\n');
  }
  return 0;
}
