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
    for (int i = 0; i < index; i++) {
        p = p->next;
    } // p is just at the list[index]
    
    new->prev = p->prev;
    new->next = p;
    p->prev = new;
    return list;
}


doublelinklist remove_dl(doublelinklist list, int index) {

}


doublelinklist destroy_dl(doublelinklist list) {


}


doublelinklist display_dl(doublelinklist list) {
    while (list != NULL) {
        printf("%-4d", list->data);
        list = list->next;
    }
    puts("");
}
