#include <stdio.h>
#include "list.h"
#include "xmalloc.h"


/**
 * @brief Push data into list
 *
 * @param list, the list or NULL
 * @param data, the data 
 * @return a new list
 *   @retval new, new head pointer
 */
conscell* lpush(conscell* list, void* data)
{
    conscell* new = xmalloc(sizeof(conscell));
    new->data = data;
    new->next = list;
    return new;
}

/**
 * @brief Pop up data 
 *
 * @param list, the list or NULL
 * @param cons, hold the popped cons
 * @return a new list
 *   @retval new, new head pointer
 */
conscell* lpop(conscell* list, conscell** cons)
{
    if (list == NULL) {
        return NULL;
    }
    // seperate list -> (top, rest)
    conscell* top = list;
    list = list->next;

    top->next = NULL;   // cut up connection between top and rest
    *cons = top;    // return popped cons
    return list;
}



void lfree(conscell* list)
{
    while (list != NULL) {
        conscell* p = list->next;
        free(list);
        list = p;
    }
}

conscell* lreverse(conscell* list)
{
    conscell* new = NULL;
    while (list != NULL) {
        // seperate list as (top, list)
        conscell* top = list;
        list = list->next;
        // make top -> new
        top->next = new;
        new = top;
    }
    return new;
}