/* Shell sort 
1. divide the array into several groups
2. directly insert in every groups
3. directly insert in final groups

    0   1   2   3   4   5   6   7   8   9
-----------------------------------------
0:  1   5   9   3   8   2  10  94  23  12
1:  
2:
*/

#include <stdio.h>


void println(int A[], int n){
  for (int i=0; i<n; i++){
    printf("%3d", A[i]);
  }
  puts("");
}


void insertSort(int A[], int n){
  int i, j, t;
  for (i=1; i<n; i++){
    // save part
    t = A[i]; 
    // move part
    for (j=i; j>0 && (A[j-1] > t); j--){
      A[j] = A[j-1];
    }
    // store part
    A[j] = t;
  }
}


/* d is the distance, initializes as half of length
 * every iteration, d -= 2 until d < 1 then stop
 * when d == 1, it is insert-sort
 */
void shellSort(int A[], int n){
  int t, j;

  for (int d=5; d >= 1; d -= 2){
    for (int i=d; i < n; ++i){
      // if-part is a small insert sort part
      if (A[i] < A[i-d]){
        t = A[i];
        for (j=i; j>0 && (t<A[j-d]); j -= d){
          A[j] = A[j-d];
        }
        A[j] = t;
      }
    }
  }
}

void main(){
  int A[] = {11, 5, 19, 3, 8, 2, 10, 4, 23, 12};
  int n = sizeof(A)/sizeof(int);

  println(A, n);

  shellSort(A, n);
  /* insertSort(A, n); */
  println(A, n);

}
