#include <stdio.h>

void strictlyAsc(int lst[], int len){
  int flag = 1;
  for (int i=0; i<len-1; i++)
    if (lst[i]>=lst[i+1]){
      printf("not strictly ascending!\n");
      flag = -1;
    }
  if (flag == 1)
    puts("Strictly ascending!");
}


int main(){
  
  int lst[] = {5, 6, 7, 9, 11, 14};
  int lst_[] = {5, 5, 7, 9, 11, 15};
  int len = 6;
  
  strictlyAsc(lst, len);
  strictlyAsc(lst_, len);

  
  return 0;
}
