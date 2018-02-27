/* #define WARNING "Mory English" */
#include <stdio.h>


#define MAXSIZE 20             
#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

typedef int Status;
typedef int ElemType;

typedef struct {
  ElemType data[MAXSIZE];
  int length;
}seqList;


Status InitElem(seqList *L){
  L->length = 0;
  return OK;
}



Status GetElem(seqList L, int i, ElemType *e) {
  
  if (L.length == 0 || i < 0 || i > L.length-1)
    return ERROR;                /* something wrong */
  
  *e = L.data[i];                /* get the value */
  return OK;                     /* success */
}


Status InsertElem(seqList *L, int i, ElemType e) {

  if ((L->length == MAXSIZE) || (i < 0) || (i > MAXSIZE-1)) 
    return ERROR;               /* something wrong */

  while (i > L->length)         // shift i
    i--;

  for (int k = i; k < L->length; k++)
    L->data[k+1] = L->data[k];
  L->data[i] = e;               /* insert value */
  L->length++;                  /* increse length */
  return OK;
}


Status DeleteElem(seqList *L, int i) {
  if (i < 0 || i >= L->length || L->length == 0)
    return ERROR;
  
  for (int k = i; k < L->length; k++)
    L->data[k] = L->data[k+1];

  L->length--;
  return OK;
}


Status PrintElem(seqList L){
  printf("Elements in List: ");
  for (int i=0; i<L.length; i++)
    printf("%d\t", L.data[i]);
  puts("");
  return OK;
}


void main(){

  seqList list;
  InitElem(&list);

  InsertElem(&list, 5, 9);
  InsertElem(&list, 0, 34);
  InsertElem(&list, 1, 42);
  InsertElem(&list, 2, 13);
  PrintElem(list);

  DeleteElem(&list, 0);
  PrintElem(list);

  ElemType e;
  GetElem(list, 1, &e);
  printf("the 1st element is: %d\n", e);
  
}


/* Test Note:

   Mory version. if insert in arbitray position and cause empty between 
   data, the data will append to the end of the list

 */
