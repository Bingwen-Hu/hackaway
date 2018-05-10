#include "doubly_linklist.h"
#include <stdio.h>

int main() {
    doublelinklist list = NULL;
    
    for (int i = 0; i < 10; i++) {
        list = insert_dl(list, i, i*i);
    }
    display_dl(list);
    destroy_dl(list);
}