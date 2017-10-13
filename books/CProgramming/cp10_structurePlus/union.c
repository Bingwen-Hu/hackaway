#include <stdio.h>

union data{

  char c;
  int i;
  long l;
  float f;

} d;


void main(){


  d.c = 'M';
  d.i = 10;
  d.f = 54.3;
  d.l = 66666;

  printf("d is %d", d.l);


}
