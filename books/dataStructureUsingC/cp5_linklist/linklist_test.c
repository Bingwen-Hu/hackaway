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

    puts("\ntest insert at");
    list = insert_at(list, 2, 42);
    display(list);

    puts("\ntest delete begin");
    list = delete_begin(list);
    display(list);

    puts("\ntest delete end");
    list = delete_end(list);
    display(list);

    puts("\ntest delete at");
    list = delete_at(list, 1);
    display(list);

    puts("\ntest delete at");
    list = delete_at(list, 0);
    display(list);


    puts("\ntest sort");
    list = insert_at(list, 3, 99);
    list = insert_at(list, 5, 49);
    list = sort(list);
    display(list);

    destroy(list);
}