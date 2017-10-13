#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct emp{
  
  int len;
  char *name;

} emp, *empptr;


void main(){

  char newname[] = "Mory";
  /* struct emp *p = (struct emp *) malloc(sizeof(struct emp)); */
  empptr p = (empptr) malloc(sizeof(emp));

  p->len = strlen(newname);
  p->name = malloc(p->len)+1;
  strcpy(p->name, newname);
  printf("%d %s", p->len, p->name);
  puts("");

  printf("the address of p: %u\n", &p);
  printf("the address of the object: %u\n", p);

  free(p->name);
  free(p);

}


/* Test Note:

   KEYWORD KEYWORD NAME BODY NICNAME
   typedef struct  stru {}   stru, *struptr;

   so, define:
   >>> stru s    -> struct stru s;
   >>> struptr p -> struct stru *p;
   
   cast:
   >>> (struptr) -> (struct stru *)

   tricks:
   `struct stru` is a whole word, different from stru.
   


   <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
   malloc with Structure.

   structure need space itself and its pointer needs a 
   whole space to pointer to.
   
   >>> struptr p = (struptr) malloc(sizeof(stru));
   
   fields in stucture have allocate space now, if there is any pointer, 
   the pointer itself have a space, but it values is still NULL.

   so, the address of the pointer is not the address of the
   object it pointers to.

   In array, it's like a pointer, but it is different at all.
   
 */
