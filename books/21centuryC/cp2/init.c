#include <stdio.h>

int main(){
  
  int isprime[] = {[1]=1, 1,1, [7]=1, [11]=1};

  for (int i=0; i<12; i++) {
    printf("element %i: %i\n", i, isprime[i]);
  }
  
  return 0;
}
