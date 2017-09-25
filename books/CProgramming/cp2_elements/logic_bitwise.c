#include <stdio.h>

int main() {

  int a = 23;
  int b = 10;

  unsigned int c = 23 | 10;
  
  printf("bitwise operator\n");
  printf("a | b is : %u\n", c);
  printf("a & b is : %u\n", a & b);
  printf("a ^ b is : %u\n", a ^ b);
  
  printf("logical operator\n");
  printf("a && b is: %u\n", a && b);
  printf("a || b is: %u\n", a || b);
  printf("!a is : %u\n", !a);
  
  return 0;
}
