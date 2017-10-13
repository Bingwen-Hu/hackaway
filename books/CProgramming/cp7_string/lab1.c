#include <stdio.h>
#include <string.h>


void main(){

  char s1[40] = "Mory";
  char s2[10] = "Ann";

  strcat(s1, " ");
  strcat(s1, s2);


  printf("%s", s1);


}
