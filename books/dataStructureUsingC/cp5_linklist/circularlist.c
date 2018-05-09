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
void display(circularlist list) {
    circularlist p = list->next;
    while (p->next != list->next) {
        printf("%-4d", p->data);
        p = p->next;
    }
    puts("\n");
}
void destroy(circularlist list);
