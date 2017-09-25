#include <stdio.h>

int main() {

  int a, b, c;

  printf("Input three numbers:");
  scanf("%d %d %d", &a, &b, &c);
  
  int max;
  if ((a >= b) && (a >= c))
    max = a;
  else if ((b >= a) && (b >= c))
    max = b;
  else
    max = c;

  printf("the maxinum is %d", max);
  return 0;

}
