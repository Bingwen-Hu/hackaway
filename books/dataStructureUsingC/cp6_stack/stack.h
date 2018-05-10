/* normal array stack */
#pragma once

#define STACK_SIZE 200

typedef struct stack {
    int data[STACK_SIZE];
    int top;
} stack;

stack make();
int peek(stack *s);
int push(stack *s, int value);
int pop(stack *s);