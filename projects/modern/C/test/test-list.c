#include "list.h"
#include <stdio.h>


int main()
{
    conscell* list = NULL;
    int a = 101, b = -45, c = 1000, d = 12;
    list = lpush(list, &a);
    list = lpush(list, &b);
    list = lpush(list, &c);
    list = lpush(list, &d);
    lprint("%d ", list, int);

    int* cc = list->data;
    list = lpop(list);
    printf("value of c is %d\n", *cc);
    lprint("%d ", list, int);
}