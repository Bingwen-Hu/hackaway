#include <stdio.h>

int main(){

  int a[] = {10, 2, 39, 4, 58};


  printf("a is: %u\t &a is %u\n", a, &a);

  printf("a+1 is %u\t &a+1 is %u\n", a+1, &a+1);

  return 0;
}


/* Test NOTE 

   name of array is the address of the first element
   and &a is the address of the array.

   if we assume the beginning of address is 0, then
   
   a -> 0, &a -> 0
   a+1 is:
       n = sizeof(a[i]);
       a+1 -> a+n;

   &a+1 is:
       n = sizeof(a[i]);
       len = number of element in a;
       &a+1 -> a + n * len;
*/

