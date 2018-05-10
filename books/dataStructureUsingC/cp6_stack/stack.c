#include "stack.h"
#include <stdio.h>

stack make() {
    stack s = {
        .top = -1, 
        .data = {0}
    };
    return s;
}

int peek(stack *s) {
    if (s->top < 0) {
        puts("Error! stack is empty!");
        return -1;
    }
    return s->data[s->top];
}

int push(stack *s, int value) {
    if (s->top == (STACK_SIZE-1)) {
        puts("Error! stack is full!\n");
        return -1;
    }
    s->top++;
    s->data[s->top] = value;
    return 1;
}

int pop(stack *s) {
    if (s->top < 0) {
        printf("Error! stack is empty!\n");
        return -1;
    }
    int value = s->data[s->top];
    s->top--;
    return value;
}