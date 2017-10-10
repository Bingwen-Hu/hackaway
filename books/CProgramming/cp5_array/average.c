#include <stdio.h>

int main(){

  int a[5];

  for(int i=0; i<5; i++){
    scanf("%d", &a[i]);
  }
  
  int sum=0;
  double average;
  for (int i=0; i<5; i++){
    sum += a[i];
  }

  average = sum / 5.;
  printf("the average %.3f\n", average);

  return 0;
}



/* Test Note:
   
   an Integer cannot not hold a float, or something strange will happen

 */
