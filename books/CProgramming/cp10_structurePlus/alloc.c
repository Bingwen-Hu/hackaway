#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct emp{
  
  int len;
  char *name;

};


void main(){

  char newname[] = "Mory";
  struct emp *p = (struct emp *) malloc(sizeof(struct emp));

  p->len = strlen(newname);
  p->name = malloc(p->len)+1;
  strcpy(p->name, newname);
  printf("%d %s", p->len, p->name);
  free(p->name);
  free(p);

}
