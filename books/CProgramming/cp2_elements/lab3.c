#include <stdio.h>

int main(){
  
  int input;
  int reverse = 0;
  int temp;
  
  printf("Please input a number\n");

  scanf("%d", &input);

  while (input > 0) {
    
    temp = input % 10;
    reverse = reverse * 10 + temp;
    input /= 10;
  }
  
  printf("The reverse is: %d\n", reverse);
  
  return 0;
}
