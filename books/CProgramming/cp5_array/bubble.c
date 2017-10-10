/* what's bubble?  

think about it! who heavy who die......

origin: 10 4 2 7 3 1 9
1 ----: 4 2 7 3 1 9 10
2 ----: 2 4 3 1 7 9 10
3 ----: 2 3 1 4 7 9 10
4 ----: 2 1 3 4 7 9 10
5 ----: 1 2 3 4 7 9 10

it seems that there is a break hole behind the queue..........
WHO HEAVY WHO DIE!
*/

#include <stdio.h>

int main(){

  int lst[] = {10, 4, 2, 7, 3, 1, 9};
  int n = 7, temp;
  for (int i=0; i<n; i++){
    for (int j=0; j<n-i-1; j++){
      if (lst[j]>lst[j+1]){
        temp = lst[j];
        lst[j] = lst[j+1];
        lst[j+1] = temp;
      }
    }
    printf("%dth time: ", i);
    for (int k=0; k<n; k++)
      printf("%d ", lst[k]);
    putchar('\n');
  } /* loop for times */
}

/* Test Note:
   
   In printf, the type is very important and can lead to strange error!

*/
