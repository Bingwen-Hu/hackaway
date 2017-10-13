#include <stdio.h>

struct student{

  int roll;
  char grade;
  int marks;

};

void main(){

  struct student s1 = {2, 'A', 93};
  
  printf("Address of roll=%u\n", &s1.roll);
  printf("Address of grade=%u\n", &s1.grade);
  printf("Address of marks=%u\n", &s1.marks);

  
}
