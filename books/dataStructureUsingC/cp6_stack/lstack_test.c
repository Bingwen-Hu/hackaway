#include "lstack.h"
#include <stdio.h>
#include <stdlib.h>


int main() {
    lstack *s = NULL;
    s = push(s, 1);
    s = push(s, 2);
    s = push(s, 42);
    
    printf("test peek: %s\n", peek(s) == 42 ? "ok" : "not ok");

    int out;
    s = pop(s, &out);
    printf("test pop: %s\n", out == 42 ? "ok" : "not ok");
    s = pop(s, &out);
    printf("test pop: %s\n", out ==  2 ? "ok" : "not ok");
    s = pop(s, &out);
    printf("test pop: %s\n", out ==  1 ? "ok" : "not ok");
    s = pop(s, &out);
    printf("test pop: %s\n", out == -1 ? "ok" : "not ok");

    destroy(s);
}