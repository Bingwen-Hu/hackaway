#include <stdio.h>
#include <stdlib.h>

#define MAXSIZE 10


typedef int ElemType;
typedef struct {

  ElemType data[MAXSIZE];
  int top;

} stack;

void Init(stack *s, int length){
  int r;
  s->top = 0;
  
  if (length > MAXSIZE){
    puts("Invalid length!");
    return;
  }

  for (int i=0; i<length; i++){
    srand(i);
    r = rand();
    s->data[i] = r;
    s->top++;
  }
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
    printf("%-4d", s.data[i]);
  }
  puts("");
}


int Pop(stack *s){
  int v;
  s->top--;
  v = s->data[s->top];
  return v;
}


void main(){
  stack s;
  Init(&s, 10);


  Print(s);

  int v = Pop(&s);
  printf("top value is %d\n", v);

  Print(s);
  
}
