#include <stdio.h>

struct employee{
 
  int gender:1;
  int marry_status:2;
  int hobbies:3;
  int scheme:4;

};


void main(){

  struct employee e = {0, 1, 4, 8};

  printf("values of gender: %u\n", e.gender);
  printf("values of marry_status: %u\n", e.marry_status);
  printf("values of hobbies: %u\n", e.hobbies);
  printf("values of scheme: %u", e.scheme);

  
  
}


/* Test Note:
   
   Yes, bit fields is very small......

 */
