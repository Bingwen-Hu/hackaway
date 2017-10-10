/* Select sort

look over all the element and select the smallest and swap
very intuitive.

 */


#include <stdio.h>

#define SIZE 20

int k_smallest(int k, int a[], int len);

int main(){

  int a[SIZE];

  puts("Enter 20 numbers:");
  for (int i=0; i<SIZE; i++){
    printf("%d number: ", i+1);
    scanf("%d", &a[i]);
  }

  int k;
  puts("Enter k represent which one you want...");
  scanf("%d", &k);

  int k_small = k_smallest(k, a, SIZE);

  printf("%dth smallest number is %d", k, k_small);

  return 0;
}

int k_smallest(int k, int a[], int len){

  int temp, i;
  for (i=0; i<len-1; i++){
    for (int j=i+1; j<len; j++){
      if (a[i] == a[j]){
        temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
    if (i == (k-1))
      break;
  }
  return a[i];
}


/* Test Note:

   sorting is annoy......
      miss for elegance...
 */
