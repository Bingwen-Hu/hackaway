#include <stdio.h>
#include <string.h>


int str2int(char string[]);


void main(){

  char string[] = "124";
  
  int num = str2int(string);
  
  printf("the number is: %d", num);

}


int str2int(char string[]){
  
  int len = strlen(string);
  int num = 0, temp;
  for (int i=0; i<len; i++){
    temp = (int)string[i] - (int)'0';
    printf("%d ", temp);
    num = num * 10 + temp;

  }
  puts("");
  return num;
}
