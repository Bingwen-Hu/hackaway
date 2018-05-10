#include "stack.h"
#include <stdio.h>

int main() {
    stack s = make();

    push(&s, 3);
    push(&s, 5);
    push(&s, 2);

    printf("test stack\n");
    printf("pop one value...%s\n", pop(&s) == 2 ? "ok" : "not ok");
    printf("pop one value...%s\n", pop(&s) == 5 ? "ok" : "not ok");
    printf("pop one value...%s\n", pop(&s) == 3 ? "ok" : "not ok");
    printf("test empty stack...%s\n", pop(&s) == -1 ? "ok" : "not ok");
}