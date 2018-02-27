#include <stdio.h>
#include <stdlib.h>


typedef struct StackNode {
  int data;
  struct StackNode *next;
}StackNode;

typedef struct LinkStack{
  StackNode *top;
  int count;
}LinkStack;

void init(LinkStack *stack){
  stack->count = 0;
  stack->top = NULL;
}

void push(LinkStack *stack, int value){
  StackNode *node = (StackNode *)malloc(sizeof(StackNode));
  node->data = value;
  node->next = stack->top;
  stack->top = node;
  stack->count++;
}

void print(LinkStack *stack){
  StackNode *p = stack->top;

  printf("Stack: ");
  for (int i=0; i<stack->count; i++){
    printf("%-3d", p->data);
    p = p->next;
  }
  puts("");

}

int pop(LinkStack *stack){
  int data;
  StackNode *p;
  
  p = stack->top;  
  data = p->data;
  stack->top = stack->top->next;
  stack->count--;

  free(p);
  return data;
}

void destroy(LinkStack *stack){
  StackNode *p = stack->top;
  StackNode *q;

  for (int i=0; i<stack->count; i++){
    q = p;
    p = p->next;
    free(q);
  }
  free(stack);
}

void main(){
  
  LinkStack *stack = (LinkStack *)malloc(sizeof(LinkStack));
  init(stack);
  push(stack, 10);
  push(stack, 20);
  push(stack, 42);
  print(stack);

  int data = pop(stack);
  printf("the top data: %d\n", data);
  print(stack);

  destroy(stack);
}

