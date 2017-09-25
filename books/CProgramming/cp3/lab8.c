#include <stdio.h>

int inputInt(char *msg){

  int i;
  printf(msg);
  scanf("%d", &i);
  return i;
}


int main(){

  int i = inputInt("Input an integer: ");
  
  int grade = 'O';

  if (i >= 90)
    grade = 'A';
  else if (i >= 70)
    grade = 'B';
  else if (i >= 50)
    grade = 'C';
  else 
    grade = 'F';
  
  printf("Your grade is %c", grade);

  return 0;
}
