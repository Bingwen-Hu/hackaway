#include "lqueue.h"
#include <stdio.h>


int main(int argc, char const *argv[])
{
    lqueue *q = create_lq();
    int value;
    q = insert_lq(q, 5);
    q = insert_lq(q, 12);
    q = insert_lq(q, 13);
    display_lq(q);
    q = remove_lq(q, &value);
    q = insert_lq(q, 289);
    display_lq(q);
    destroy_lq(q);

    return 0;
}
