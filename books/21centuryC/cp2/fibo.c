#include <math.h>
#include <stdio.h>

int main(){

  int pre1 = 0;
  int pre2 = 1;
  
  int next;
  double ratio;

  for (int i=0; i<20; i++) {
    next = pre1 + pre2;
    ratio = (double) pre2 / next;
    printf("%ith times, ratio is %.6f\n", i, ratio);

    pre1 = pre2;
    pre2 = next;
  }

  return 0;
}
