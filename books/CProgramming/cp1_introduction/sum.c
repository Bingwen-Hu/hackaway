#include <stdio.h>              /* for printf, scanf */
#include <conio.h>              /* for clrscr, getch */

int main(){
  
  int a, b, sum;

  /* clrscr(); */
  printf("enter two numbers, comma as seperators\n");
  scanf("%d,%d", &a, &b);
  sum = a + b;

  printf("sum=%d\n", sum);
  getch();
}
