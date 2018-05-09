#include "circularlist.h"
#include <stdio.h>


int main() {
    circularlist list = make();

    puts("test insert begin");
    for (int i=0; i < 10; i++) {
        insert_begin(list, i);
    }
    display(list);
    destroy(list);
}