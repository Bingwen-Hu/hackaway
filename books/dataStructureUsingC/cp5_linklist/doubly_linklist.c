#include "doubly_linklist.h"
#include <stdlib.h>
#include <stdio.h>



doublelinklist insert_dl(doublelinklist list, int index, int value) {
    node *new = malloc(sizeof(node));
    new->data = value;

    if (list == NULL) {
        new->prev = NULL;
        new->next = NULL;
        return new;
    } 

    doublelinklist p = list;
    for (int i = 0; i < index-1; i++) {
        p = p->next;
    } // p is just previous to  the list[index]
    
    new->prev = p;
    new->next = p->next;
    p->next = new;
    return list;
}


doublelinklist remove_dl(doublelinklist list, int index) {

}


void destroy_dl(doublelinklist list) {
    doublelinklist p = list;
    while (list != NULL) {
        p = list;
        list = list->next;
        free(p);
    }
}


void display_dl(doublelinklist list) {
    while (list != NULL) {
        printf("%-4d", list->data);
        list = list->next;
    }
    puts("");
}
