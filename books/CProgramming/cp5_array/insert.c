/* Insert sort

Every time we insert a new element to a new array.... new array?

origin: 10 4 2 7 3 1 9
1 ----: 4 10 2 7 3 1 9
2 ----: 2 4 10 7 3 1 9
3 ----: 2 4 7 10 3 1 9
4 ----: 2 3 4 7 10 1 9
5 ----: 1 2 3 4 7 10 9
6 ----: 1 2 3 4 7 9 10

At the very beginning, the first element is a sorted array, 
and we insert the rest element in it.

*/


#include <stdio.h>

int main(){
  
  int lst[] = {10, 4, 2, 7, 3, 1, 9};
  int n = 7, temp;
  
  for (int i=1; i<n; i++){
    temp = lst[i];
    for (int j=i-1; j>=0; j--){
      if (lst[i]<lst[j]){
        lst[j+1] = lst[j];
        lst[j] = temp;
      } 
    }

    printf("%dth time: ", i);
    for (int k=0; k<n; k++)
      printf("%d ", lst[k]);
    putchar('\n');

  }
  

  return 0;
}



/* Test Note:

   when we move back, the index is from high to low

 */
