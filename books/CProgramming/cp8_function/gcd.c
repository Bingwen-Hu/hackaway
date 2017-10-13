#include <stdio.h>

int gcd(int a, int b);

void main(){

  int a = 42, b = 7;
  
  printf("gcd of %d and %d is: %d", a, b, gcd(a, b));
}


int gcd(int a, int b){

  if (a < b){
    a = a + b;
    b = a - b;
    a = a - b;
  }

  if ((a % b) == 0)
    return b;
  return gcd(b, a % b);
}
