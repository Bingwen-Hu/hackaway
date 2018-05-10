/* multistack, namely double stack */
#include "multistack.h"

void initstack(mstack *s) {
    s->topA = -1;
    s->topB = STACK_SIZE;
}

int peekA(mstack *s) {
    return s->data[s->topA];
}

int peekB(mstack *s) {
    return s->data[s->topB];
}

int pushA(mstack *s, int value) {
    int all = s->topA + STACK_SIZE - s->topB;
    if (all > STACK_SIZE) {
        puts("Error! stack is full!");
        return -1;
    }
    s->topA++;
    s->data[s->topA] = value;
    return 1;
}

int pushB(mstack *s, int value) {
    int all = s->topA + STACK_SIZE - s->topB;
    if (all > STACK_SIZE) {
        puts("Error! stack is full!");
        return -1;
    }
    s->topB--;
    s->data[s->topB] = value;
    return 1;
}


int popB(mstack *s) {
    int value = s->data[s->topB];
    s->topB++;
    return value;
}
int popA(mstack *s) {
    int value = s->data[s->topA];
    s->topA++;
    return value;
}
