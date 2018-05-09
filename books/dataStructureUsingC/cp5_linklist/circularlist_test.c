#include "circularlist.h"
#include <stdio.h>


int main() {
    circularlist list = make();

    puts("test insert begin");
    for (int i=0; i < 10; i++) {
        insert_begin(list, i);
    }
    display(list);

    puts("test insert end");
    insert_end(list, 12);
    insert_end(list, 25);
    display(list);


    puts("test delete begin");
    delete_begin(list);
    delete_begin(list);
    display(list);

    puts("test delete end");
    delete_end(list);
    delete_end(list);
    display(list);

    destroy(list);
}