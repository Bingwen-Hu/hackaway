#include <stdio.h>


void main(){

  char ch;
  int pre = 0;
  while((ch=getchar())!='\n'){
    if (ch != ' '){
      printf("%c", ch);
      pre = 0;
    } else if (pre == 0){
      printf("%c", ch);
      pre = 1;
    }
  }

}
