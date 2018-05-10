#pragma once

typedef struct node {
    int data;
    struct node *next;
} node, lstack;

int peek(lstack *s);
lstack *push(lstack *s, int value);
lstack *pop(lstack *s, int *value);
void destroy(lstack *s);

