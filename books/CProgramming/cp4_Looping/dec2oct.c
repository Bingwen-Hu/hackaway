#include <stdio.h>
#include <math.h>

int main(){

  int n;
  printf("Enter an integer: ");
  scanf("%d", &n);
  

  int remain, result=0, cnt=0;
  while (n > 0) {
    remain = n % 8;
    n = n / 8;
    result += remain*pow(10, cnt);
    cnt++;
  }

  printf("The octal equivalent is %d", result);
      
  return 0;
}
