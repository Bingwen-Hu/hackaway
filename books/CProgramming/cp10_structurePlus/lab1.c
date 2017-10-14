#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct employer {

  int number;
  char *name;
  
} employer, *employerPtr;

void main(){

  employerPtr emps = (employerPtr) malloc(sizeof(employer));

  int number; 
  char name[20];

  puts("Enter employee's number and name");
  scanf("%d %s", &number, name);
  
  emps->number = number;

  int len = strlen(name);
  emps->name = malloc(len);
  strcpy(emps->name, name);
  
  printf("%dth employer named %s\n", emps->number, emps->name);

  free(emps->name);
  free(emps);

}
