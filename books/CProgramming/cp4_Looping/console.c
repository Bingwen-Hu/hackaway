/* readable matters */

#include <stdio.h>


int main(){
  
  int count=0;

  float num=0, sum=0, avg=0;

  for(count=0; count<7; count++){
    printf("Enter temperature: ");
    scanf("%f", &num);
    sum = sum + num;
  }

  avg = sum / 7;
  printf("\nAverage=%f\n", avg);

  
  return 0;
}
