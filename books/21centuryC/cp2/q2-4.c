#include <math.h>
#include <stdio.h>


int main(){
  
  int the_array[100];

  for (int i=0; i<100; i++) {
    the_array[i] = pow(i, 2);

    printf("%i squared is %i\n", i, the_array[i]);
  }

  return 0;
}
