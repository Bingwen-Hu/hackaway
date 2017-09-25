#include <stdio.h>

int main(){
  
  char ch;
  
  while((ch=getchar()) != '\n'){
    if (ch == '.')
      continue;
    
    putchar(ch);
  }
  
  return 0;
}


/* Test Note

   1. point(.) is ignore
   2. when Enter is hit, new line is print to console.

   3. continue is the same as Python.

 */
