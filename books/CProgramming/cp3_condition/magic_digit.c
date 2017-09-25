#include <stdio.h>

int reverse(int i){
  
  int re = 0;

  while (i > 0) {
    re = re * 10 + i % 10;
    i /= 10;
  }
    
  return re;
}



int main(){

  int i;
  printf("Input an integer! ");
  scanf("%d", &i);

  int re = reverse(i);

  if (i == re)
    printf("Your number is MAGIC!");
  else
    printf("Nothing exciting Orz.....");
  
  return 0;
}
