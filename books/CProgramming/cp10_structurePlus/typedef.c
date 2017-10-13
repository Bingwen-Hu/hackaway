#include <stdio.h>

/* typedef `struct s` -> `s` */
typedef struct node{

  int value;
  struct node *node;

} node, *node;


void main(){

  node n = {1, NULL};


  printf("value of n is %d", n->value);



}

