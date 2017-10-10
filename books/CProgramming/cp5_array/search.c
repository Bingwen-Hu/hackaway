#include <stdio.h>

#define SIZE 9

int linearSearch(int a[], int len, int key);
int binarySearch(int a[], int len, int key);


int main(){

  int niddle, index = -1;
  int haystack[SIZE] = {1, 3, 5, 6, 7, 9, 14, 23, 43};

  printf("Which number do you want? ");
  scanf("%d", &niddle);
  
  
  /* index = linearSearch(haystack, SIZE, niddle);  */
  index = binarySearch(haystack, SIZE, niddle);
  if (index >= 0){
    printf("Found it! index is %d\n", index);
  } else {
    printf("Niddle %d is not found\n", niddle);
  }

  return 0;
}


int linearSearch(int a[], int len, int key){

  int index = -1;
  for (int i=0; i<len; i++){
    if (a[i]==key)
      index = i;
  }
  return index;
}


int binarySearch(int a[], int len, int key){

  int index = -1;
  int half = len/2, start=0, end=len;
  while (start <= end){
    if (a[half]==key){
      printf("debug >>> found it, half is: %d\n", half);
      index = half;
      break;
    } else if (a[half]<key){
      start = half+1;
      printf("debug >>> right up, start is: %d\n", start);
    } else {
      end = half-1;
      printf("debug >>> left down, end is: %d\n", end);
    }
    half = (end+start) / 2;

    for (int i=start; i<end; i++)
      printf("%d ", a[i]);
    putchar('\n');
  }

  return index;
}



/* Test Note:

   Let my honor debug message there!
   Stuck in for more than one hour ......
   just miswrite the '<' as '>'

 */
