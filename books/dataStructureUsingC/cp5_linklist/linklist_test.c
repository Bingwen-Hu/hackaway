#include "linklist.h"
#include <stdio.h>

int main() {
    linklist list = make();
    display(list);
    
    for (int i = 0; i < 10; i++) {
        list = insert_begin(list, i);
    }
    display(list);

    puts("\ntest insert end");
    list = insert_end(list, 201);
    display(list);


    destroy(list);
}