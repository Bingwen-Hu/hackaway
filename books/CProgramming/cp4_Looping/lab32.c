#include <stdio.h>


int main(){

  double d = 0, sum = .0;
  int count = 0;
  
  printf("input numbers, -1 to terminate: ");
  scanf("%lf", &d);

   while(d != -1){
     printf("input numbers, -1 to terminate: ");
     count++;
     sum += d;
     scanf("%lf", &d);
   }
  printf("the average is %.3lf", sum/count);
  
  return 0;
}




/* Test Note:
   
   In fact, we code for mistake, not for knowledge

 */
