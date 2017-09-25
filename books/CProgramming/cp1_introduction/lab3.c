#include <stdio.h>

/* typedef (void)*(lab7(int *a, int *b)) LAB7 */

void lab7(int *a, int *b){
  /* swap two number without using third variable */
  *a = *a + *b;                 /* get the sum */
  *b = *a - *b;                 /* get the origin a */
  *a = *a - *b;                 /* get the origin b */
}

  
void printLab(LAB7 lab7, int a, int b){
  printf("init: a=%i, b=%i\n", a, b);
  lab7(&a, &b);
  printf("last: a=%i, b=%i\n", a, b);
}


int main(){
  /* 
  int a, b;

  printf("Enter two number, must be integer!\n");
  scanf("%i %i", &a, &b);
  
  printf("Average of two integers %i and %i is: %.3f\n", a, b, (a+b)/2.0);
  return 0;
  */
  int a = 2;
  int b = 5;

  /* test for lab7 */
  printf("init: a=%i, b=%i\n", a, b);
  lab7(&a, &b);
  printf("last: a=%i, b=%i\n", a, b);
  
  /* printLab(); */
  
  return 0;
  
}





/* test NOTE: */
/* If inputs are not integer, something weird will happen
 * scanf using a pattern to match the user input

 * Another note: how to use a function pointer?
 */
