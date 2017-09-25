#include <stdio.h>


int main(){

  int n, fact=1;
  printf("Input an integer ");
  scanf("%d", &n);

  for (int i=1; i<=n; i++) {
    fact *= i;
  }

  printf("%d! = %d", n, fact);

  return 0;
}
