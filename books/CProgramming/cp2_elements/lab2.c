#include <stdio.h>


/* Merge three number eg. a=1, b=2, c=8, ==> 128 */
int main(){
  
  int i = -1, sum = 0;
  
  printf("Input numbers 0-9, input -1 to exit \n");
  
  do {
    scanf("%i", &i);
    sum = sum * 10 + i;
    printf("sum is : %d\n", sum);
  } while(i != -1);

  printf("Goodbye...\n");
  return 0;
}
