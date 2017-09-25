#include <stdio.h>

int main(){

  int i = 0; 

  printf("input an integer:");
  scanf("%d", &i);


  printf("You have input an %s", i % 2 == 0 ? "even" : "odd");

  return 0;
}
