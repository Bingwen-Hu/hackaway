#include <stdio.h>

/* calcute the sum of digits of a three digit number
   eg. 125 => 8
 */


int main(){

  int i;
  printf("Input a digit\n");
  scanf("%d", &i);
  
  int sum = 0;
  while (i > 0){
    
    sum += i % 10;
    i /= 10;
  }
  printf("sum of digits is: %d", sum);

  return 0;
}
