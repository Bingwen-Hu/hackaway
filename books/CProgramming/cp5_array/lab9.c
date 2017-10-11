/* read an array of integers and print even and odd separately */

#include <stdio.h>

void printByParity(int lst[], int len);

int main(){
  
  int lst[] = {2, 3, 6, 8, 11, 18, 21, 25};
  int len = 8;
  printByParity(lst, len);

  return 0;
}



void printByParity(int lst[], int len){
  int even[len], odd[len], nEven, nOdd;
  nEven = nOdd = 0;
  for (int i=0; i<len; i++){
    if (lst[i] % 2 == 0){
      even[nEven] = lst[i];
      nEven++;
    } else {
      odd[nOdd] = lst[i];
      nOdd++;
    }
  }
  printf("Even numbers: ");
  for (int i=0; i<nEven; i++){
    printf("\t%d", even[i]);
  }
  puts("");
  printf("Odd numbers: ");
  for (int i=0; i<nOdd; i++){
    printf("\t%d", odd[i]);
  }
  puts("");
    
}
