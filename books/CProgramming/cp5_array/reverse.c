#include <stdio.h>

int main(){

  int a[] = {49, 48, 48, 23, 4, 96, 48, 2};
  int n = 8;
  puts("The origin array: ");
  for (int i=0; i<n; i++)
    printf("%d ", a[i]);

  
  puts("\nThe reverse array: ");
  for (int i=n-1; i>=0; i--)
    printf("%d ", a[i]);

  return 0;
}
