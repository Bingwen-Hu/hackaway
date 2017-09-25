#include <stdio.h>

int main(){
  
  int i, n, r, f=0;
  printf("Please input an integer: ");
  scanf("%d", &i);


  for (int p=15; p>0; p--) {
    
    n = 1<<p;
    r = n&i;
    if (f && (r == 0))
      putchar('0');
    else if (r > 0){
      putchar('1');
      f = 1;
    }
  }
    
  return 0;
}


/* Test Note

   Hard Way is the easies way.
   The easiest problem is the hardest problem!
   
 */
