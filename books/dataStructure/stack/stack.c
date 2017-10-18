/* using a stack to perform operation  */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXSIZE 100


typedef char ElemType;
typedef struct {
  ElemType data[MAXSIZE];
  int top;
}stack;


void Init(stack *s){
  s->top = 0;
}


void Push(stack *s, ElemType e){
  if (s->top >= MAXSIZE){
    puts("stack is full!");
  } else {
    s->data[s->top++] = e;
  }
}


void Print(stack s){
  printf("stack: ");
  for (int i=0; i<s.top; i++){
    printf("%-4c", s.data[i]);
  }
  puts("");
}


ElemType Pop(stack *s){
  ElemType v;
  s->top--;
  v = s->data[s->top];
  return v;
}


int compute(const char string[], int length){
  int i=0, cres, c1, c2;
  ElemType e;

  stack s;
  Init(&s);
  
  while(i < length){
    if ('0' <= string[i] && string[i] <= '9'){
      Push(&s, string[i]);
    } else if (string[i] == '-'){
      c1 = Pop(&s) - 48;
      c2 = Pop(&s) - 48;
      cres = c2 - c1;
      printf("%d - %d = %d\n", c2, c1, cres);
      e = cres + 48;
      Push(&s, e);
    } else if (string[i] == '+'){
      c1 = Pop(&s) - 48;
      c2 = Pop(&s) - 48;
      cres = c2 + c1;
      printf("%d + %d = %d\n", c2, c1, cres);
      e = cres + 48;
      Push(&s, e);
    } else if (string[i] == '*'){
      c1 = Pop(&s) - 48;
      c2 = Pop(&s) - 48;
      cres = c2 * c1;
      printf("%d * %d = %d\n", c2, c1, cres);
      e = cres + 48;
      Push(&s, e);
    } else if (string[i] == '/'){
      c1 = Pop(&s) - 48;
      c2 = Pop(&s) - 48;
      cres = c2 / c1;
      printf("%d / %d = %d\n", c1, c2, cres);
      e = cres + 48;
      Push(&s, e);
    } else {
      
    }
    i++;
  }
  return cres;
}


void main(){
  char string[] = "9 3 1 - 3 * + 9 2 - + ";
  int len = strlen(string);
  int res = compute(string, len);
  
  printf("the result is %d\n", res);
}
