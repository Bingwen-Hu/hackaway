#include <stdio.h>

int main() {

  int i;

  printf("How many calls do you make?");
  scanf("%d", &i);
  
  if (i < 0) {
    printf("Error, position integers are expected!");
    return 1;
  }


  double rate;
  if ((i >= 0) && (i < 100))
    rate = 0.0;
  else if ((i >= 100) && (i < 200))
    rate = 0.8;
  else if ((i >= 200) && (i < 500))
    rate = 1.00;
  else 
    rate = 1.20;

  printf("Your tel bill is %.3lf", rate * i);
    
  return 0;
}


/* Test Note

   GDB is an essential tool! I just miss a & before i in scanf statement.

 */
