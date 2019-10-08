#include "list.h"
#include <stdio.h>
#include <stdlib.h>


int main()
{
    conscell* list = NULL;
    int a = 101, b = -45, c = 1000, d = 12;
    list = lpush(list, &a);
    list = lpush(list, &b);
    list = lpush(list, &c);
    list = lpush(list, &d);
    lprint("%d ", list, int);

    // keep the first data
    conscell* cc = NULL;
   // pop it out
    list = lpop(list, &cc);
    int* data = cc->data;
    printf("value of c is %d\n", *data);
    lprint("%d ", list, int);

    printf("reverse list\n");
    list = lreverse(list);
    lprint("%d ", list, int);

    // free both list   
    lfree(list);
    lfree(cc);
}