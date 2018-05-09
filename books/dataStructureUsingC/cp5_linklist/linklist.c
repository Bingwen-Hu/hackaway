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
    printf("\n");
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

linklist insert_at(linklist list, int index, int value) {
    node *new = malloc(sizeof(node));
    new->data = value;

    if (index < 0 || index > list->data) {
        printf("index out of scope! length of list is %d, index is %d\n", list->data, index);
        return list;
    }
    linklist p = list;
    for (int i = 0; i < index; i++) {
        p = p->next;
    } // p just at the index
    new->next = p->next;
    p->next = new;
    list->data++;
    return list;
}

linklist delete_begin(linklist list) {
    linklist p = list->next;
    list->next = p->next;
    list->data--;
    free(p);
    return list;
}

linklist delete_end(linklist list) {
    linklist p, q;
    p = q = list;
    while (q->next != NULL) {
        p = q;
        q = q->next;
    }
    free(q);
    p->next = NULL;
    list->data--;
    return list;
}

linklist delete_at(linklist list, int index) {
    linklist p, q;
    p = q = list;

    for (int i = 0; i <= index; i++) {
        p = q;
        q = q->next;
    }
    p->next = q->next;
    list->data--;
    free(q);
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