#include <stdio.h>


int main(){

  int a, b, c, d;

  printf("Please input four numbers: ");

  scanf("%d %d %d %d", &a, &b, &c, &d);

  int t1 = a > b ? a : b;
  int t2 = c > d ? c : d;
  int max = t1 > t2 ? t1 : t2;

  printf("The maxinum is %d", max);

  return 0;

}
