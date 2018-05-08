#include "linklist.h"
#include <stdlib.h>
#include <stdio.h>

// head node
linklist make() {
    linklist list = malloc(sizeof(node));
    list->next = NULL;
    list->data = 0;
    return list;
}

void display(linklist list) {
    printf("length of list: %d\n", list->data);
    while (list->next != NULL) {
        list = list->next;
        printf("%-4d", list->data);
    }
}

linklist insert_begin(linklist list, int value) {
    node *new = malloc(sizeof(node));
    new->data = value;
    new->next = list->next;
    list->next = new;
    list->data++;
    return list;
}

linklist insert_end(linklist list, int value) {
    node *new = malloc(sizeof(node));
    new->data = value;
    new->next = NULL;
    
    linklist p = list;
    while (p->next != NULL) {
        p = p->next;
    }
    p->next = new;
    list->data++;
    return list;
}

void destroy(linklist list) {
    linklist p = list;
    while (p->next != NULL) {
        list = p->next;
        free(p);
        p = list;
    }
    free(p);
}

linklist sort(linklist list, int incr) {
    if (incr > 0) {
        // incre sort
    } else {
        // descr sort
    }
    return list;
}