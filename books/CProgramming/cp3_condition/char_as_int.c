#include <stdio.h>


int main(){

  int ch = 'a'+'b';

  switch(ch) {

    case 'a':
    case 'b':
      printf("you have secured a");

    case 'A':
      printf("You are confused");

    case 'b' + 'a':
      printf("YOu have secured both a and b");
  }

  return 0;

}


/* Test note:

   1. if one case statement is matched and without a break statement, 
   all other test will not be performed and all the statement will be
   executed.

   2. char is a int at all
 */
