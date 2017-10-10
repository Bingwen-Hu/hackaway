#include <stdio.h>

/* Several way to init an array */

void print_init(char string[], int a[], int len){

  printf(string);
  for (int i=0; i<len; i++)
    printf(" %ld", a[i]);
  putchar('\n');
}

int main(){

  int a1[5];
  int a2[5] = {1, 3, 4, 5, 6};
  int a3[5] = {0};
  int a4[5] = {1};
//  int a5[5] = {10,,30};         /* Error */
  int a6[] = {1, 2, 3};
  static int a7[5];

  print_init("init as a[5]: ", a1, 5);
  print_init("init as a[5]={1,3,4,5,6}: ", a2, 5);
  print_init("init as a[5]={0}: ", a3, 5);
  print_init("init as a[5]={1}: ", a4, 5);
  /* print_init("init as a[5]={10,,30}: ", a5, 5);  */ 
  print_init("init as a[]={1,2,3}: ", a6, 3);
  print_init("init as static a[5]: ", a7, 5);

  return 0;
}


/* Test Note:

   init as a[5]:  4201259 4201168 4194432 2686756 2130567168
   init as a[5]={1,3,4,5,6}:  1 3 4 5 6
   init as a[5]={0}:  0 0 0 0 0
   init as a[5]={1}:  1 0 0 0 0

   init as a[]={1,2,3}:  1 2 3
   init as static a[5]:  0 0 0 0 0

*/
