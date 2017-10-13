#include <stdio.h>




void main(){


  char ch;
  
  int numOfChar = 0;

  while((ch=getchar())!='\n'){    /* when not Enter */
    if (ch != ' ')
      numOfChar++;
  }
  printf("the number of char is: %d", numOfChar);

}




/* Test note: 

   string is difficult.

*/
