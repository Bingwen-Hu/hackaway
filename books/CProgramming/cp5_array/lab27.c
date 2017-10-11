/* move the negative elements of an array to the end */
#include <stdio.h>

void moveNegative(int lst[], int lst_[], int len){

  int cnt = 0;
  
  for (int i=0; i<len; i++){
    if (lst[i]>0){
      lst_[cnt++] = lst[i];
    }
  }
  for (int j=0; j<len; j++){
    if (lst[j]<=0){
      lst_[cnt++] = lst[j];
    }
  }
}

void main(){

  int lst[] = {5, -3, 2, 6, 8, -4, 7, -6, 9};
  int len = 9;

  int lst_[9];
  moveNegative(lst, lst_, len);
  
  for (int i=0; i<len; i++)
    printf("%d ", lst_[i]);
  
}

