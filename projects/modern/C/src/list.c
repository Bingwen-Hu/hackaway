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
 * @brief Pop up data, you are response to free hold the data
 * before you pop it!
 *
 * @param list, the list or NULL
 * @return a new list
 *   @retval new, new head pointer
 */
conscell* lpop(conscell* list)
{
    if (list == NULL) {
        return NULL;
    }
    conscell* rest = list->next;
    return rest;
}



