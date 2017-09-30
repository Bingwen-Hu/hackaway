/* #define WARNING "Mory English" */

#define MAXSIZE 20             



typedef int ElemType;

typedef struct List {

  ElemType data[MAXSIZE];
  int length;

} seqList;

int GetElem(seqList L, int i, ElemType *e) {
  
  if (L.length == 0 || i < 0 || i > L.length-1)
    return 0;                   /* something wrong */
  
  *e = L.data[i-1];             /* get the value */
  return 1;                     /* success */
}


int InsertElem(seqList *L, int i, ElemType e) {

  if ((L->length == MAXSIZE) || (i < 0) || (i > MAXSIZE-1)) 
    return 0;                  /* something wrong */

  for(int k = i; k < L->length; k++)
    L->data[k+1] = L->data[k];
  
  L->data[i] = e;               /* insert value */
  L->length++;                  /* increse length */
  return 1;
}

int DeleteElem(seqList *L, int i) {
  if (i < 0 || i >= L->length || L->length == 0)
    return 0;
  
  for (int k = i; k < L->length; k++)
    L->data[k] = L->data[k+1];

  L->length--;
  return 1;
}
