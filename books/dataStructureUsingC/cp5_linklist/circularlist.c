#include "circularlist.h"
#include <stdlib.h>
#include <stdio.h>

circularlist make() {
    node *new = malloc(sizeof(node));
    new->next = NULL;
    new->data = 0;
    return new;
}
circularlist insert_begin(circularlist list, int value) {
    node *new = malloc(sizeof(node));
    new->data = value;
    
    circularlist p = list->next;
    while (p != NULL && p->next != list->next) {
        p = p->next;
    }
    if (p == NULL) {
        p = new;
        new->next = new;
    } else {
        p->next = new;
        new->next = list->next;
    } 
    list->next = new;
    list->data++;
    return list;
}

circularlist insert_end(circularlist list, int value) {
    node *new = malloc(sizeof(node));
    new->data = value;
    
    circularlist p = list->next;
    while (p != NULL && p->next != list->next) {
        p = p->next;
    }
    if (p == NULL) {
        p = new;
        new->next = new;
        list->next = new;
    } else {
        new->next = p->next;
        p->next = new;
    } 
    list->data++;
    return list;
}


circularlist delete_begin(circularlist list);
circularlist delete_end(circularlist list);

void display(circularlist list) {
    circularlist p = list->next;
    while (p->next != list->next) {
        printf("%-4d", p->data);
        p = p->next;
    }
    printf("%-4d\n", p->data);
}
void destroy(circularlist list) {
    circularlist p, q;
    p = list->next;
    while (p->next != list->next) {
        q = p;
        p = p->next;
        free(q);
    }
    free(p);
    free(list);
}
