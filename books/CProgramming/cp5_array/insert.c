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
    for (int j=0; j<i; j++){
      if (lst[i]<lst[j]){
        temp = lst[i];
        for (int k=i-1; k>=j; k--)
          lst[k+1] = lst[k];
        lst[j] = temp;
      }
    } /* compare the element and the sorted array */

    printf("%dth time: ", i);
    for (int k=0; k<n; k++)
      printf("%d ", lst[k]);
    putchar('\n');
  } /* loop for the element to insert */
    
  
  return 0;
}



/* Test Note:

   when we move back, the index is from high to low

   Analyze the complexity:
   
   for (int i=1; i<n; i++){                  n times
    for (int j=0; j<i; j++){                 i times, i from 1 to n
      if (lst[i]<lst[j]){                    
        temp = lst[i];
        for (int k=i-1; k>=j; k--)           i-j times, from 0 to n-1
          lst[k+1] = lst[k];
        lst[j] = temp;
      }
    } 

 */
