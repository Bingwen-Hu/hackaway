#include <stdio.h>

int main(){

  int p, n, ret=1;
  printf("Input two integer, p and n ");
  scanf("%d %d", &p, &n);

  for (int i=1; i<=p; i++)
    ret *= n;

  printf("The power %d of number %d is %d", p, n, ret);

  return 0;
}
