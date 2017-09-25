#include <stdio.h>


int main(){

  for(int i=0; i<7; i++) {
    for (int j=0; j<5; j++) {
      printf("%2d x %2d = %2d   ", i, j, i*j);
      if (j == 3)
        break;
    }
    putchar('\n');
  }
  

  return 0;
}


/* TEST NOTE 
   
   In C99, break statement can only jump out the deepest loop

*/
