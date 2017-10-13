/* initial as string */

#include <stdio.h>


void main(){

  char s1[10];
  char s2[7] = {'s', 'h', 'i', 'v', 'a', 'm', '\0'};

  char s3[7] = {'s', 'h', 'i', 'v', 'a', 'm'};
  char s4[7] = "shivam";

  char s[] = {'s', 'h', 'i', 'v', 'a', 'm', '\0'};

  printf("init as s[10]: %s\n", s1);
  printf("init as s[7] = {.\\0}: %s\n", s2);
  printf("init as s[7] = {.}: %s\n", s3);
  printf("init as s[7] = \"shivam\": %s\n", s4);
  printf("init as s[] = {.\\0}: %s\n", s);

}



/* Test Note:

   Even in the string, \0 need to be encode as \\0

 */
