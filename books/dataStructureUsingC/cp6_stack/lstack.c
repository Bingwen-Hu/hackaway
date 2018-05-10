#include "lstack.h"
#include <stdlib.h>
#include <stdio.h>

int peek(lstack *s) {
    if (s == NULL) {
        puts("stack is empty!");
        return -1;
    }
    return s->data;
}

lstack *push(lstack *s, int value) {
    if (s == NULL) {
        s = malloc(sizeof(lstack));
        s->data = value;
        s->next = NULL;
        return s;
    }
    node *new = malloc(sizeof(node));
    new->next = s;
    new->data = value;
    return new;
}

lstack *pop(lstack *s, int *value) {
    if (s == NULL) {
        puts("Error, stack is empty!");
        *value = -1;
        return s;
    }
    lstack *new = s->next;
    *value = s->data;
    free(s);
    return new;
}

void destroy(lstack *s) {
    lstack *p = s;
    while (s != NULL) {
        p = s;
        s = s->next;
        free(p);
    }
}
