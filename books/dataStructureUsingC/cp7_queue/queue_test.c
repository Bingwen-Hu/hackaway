#include "queue.h"
#include <stdio.h>

int main(int argc, char const *argv[])
{
    queue q;
    init(&q);
    insert_q(&q, 42);
    insert_q(&q, 12);
    insert_q(&q, 84);
    printf("test peek ... %s\n", peek(&q) == 42 ? "ok" : "not OK");
    int rem = remove_q(&q);
    printf("test remove ... %s\n", rem == 42 ? "ok" : "not ok");
    printf("test peek ... %s\n", peek(&q) == 12 ? "ok" : "not OK");
    return 0;
}
