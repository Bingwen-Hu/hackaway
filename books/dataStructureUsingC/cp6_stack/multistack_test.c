/* multistack, namely double stack */

#include "multistack.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    mstack s;
    initstack(&s);

    pushA(&s, 12);
    pushA(&s, 42);
    printf("test PeekA ... %s\n", peekA(&s) == 42 ? "ok" : "not ok");
    printf("test popA ... %s\n", popA(&s) == 42 ? "ok" : "not ok");

    pushB(&s, 22);
    printf("test PeekB ... %s\n", peekB(&s) == 22 ? "ok" : "not ok");

    pushB(&s, 123);
    pushB(&s, 13);
    printf("test popB ... %s\n", popB(&s) == 13 ? "ok" : "not ok");


}