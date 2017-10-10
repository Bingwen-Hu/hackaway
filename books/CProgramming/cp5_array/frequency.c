#include <stdio.h>

#define SIZE 20

void parraledSort(int freq[], int a[]);

int main(){

  int a[SIZE];
  
  puts("Enter 20 numbers:");
  for (int i=0; i<SIZE; i++){
    printf("%d number: ", i+1);
    scanf("%d", &a[i]);
  }

  int b[SIZE], freq[SIZE]={0}, cnt=0, found=0;
  
  for (int i=0; i<SIZE; i++){
    for (int j=0; j<cnt; j++){
      if (b[j]==a[i]){
        found = 1;
        freq[j]++;
      }
    } /* loop for b[SIZE] */
    if (!found){
      b[cnt++] = a[i];
    }
    found = 0;
  } /* loop for a[SIZE] */
  

  return 0;
}


void paralledSort(int freq[], int a[], int cnt){
  

} /* sort the Array A according to frequency */
