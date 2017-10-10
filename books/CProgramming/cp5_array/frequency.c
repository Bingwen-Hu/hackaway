#include <stdio.h>

#define SIZE 20

void paralledSort(int freq[], int a[], int cnt);

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
      }} /* loop for b[SIZE] */
    if (!found){
      b[cnt] = a[i];
      freq[cnt++]++;
      
    }
    found = 0;
  } /* loop for a[SIZE] */

  paralledSort(freq, b, cnt);
  
  for (int i=0; i<cnt; i++){
    printf("frequency: %d, number: %d\n", freq[i], b[i]);
  }

  return 0;
}


void paralledSort(int freq[], int a[], int cnt){
  int temp;

  for (int i=0; i<cnt-1; i++){
    for (int j=i+1; j<cnt; j++){
      if (freq[i]>freq[j]){
        temp = freq[i];
        freq[i] = freq[j];
        freq[j] = temp;
        temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
  } 
} /* sort the Array A according to frequency */
