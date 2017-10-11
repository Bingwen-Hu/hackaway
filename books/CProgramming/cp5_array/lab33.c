/* replaces multiple occurrence of any element and move them to the end 
   For example:
   1 1 2 2 3 4 2 1 5 -> 1 0 2 0 3 4 0 0 5 -> 1 2 3 4 5 0 0 0 0
*/



#include <stdio.h>

#define SIZE 9

void replaceAndMove(int lst[], int lst_[], int len){

  int cnt = 0, exists = 0;
  
  for (int i=0; i<len; i++){
    for (int j=0; j<cnt; j++){
      if (lst[i]==lst_[j]){
        exists = 1;
      }
    }
    if (!exists){
      lst_[cnt++] = lst[i];
    }
    exists = 0;
  }

  for (int k=cnt; k<len; k++)
    lst_[k] = 0;
}



void main(){

  int lst[SIZE] = {1, 1, 2, 2, 3, 4, 2, 1, 5};
  int lst_[SIZE];

  replaceAndMove(lst, lst_, SIZE);

  for (int i=0; i<SIZE; i++)
    printf("%d ", lst_[i]);
  
}
