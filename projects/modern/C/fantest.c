#include "fantest.h"
// fan test

//
void array_vs_pointer_in_str(){
    char array[] = "text with width";
    char *pointer = "text with width";
    printf("size of array is %zu\n", sizeof(array));
    printf("size of pointer is %zu\n", sizeof(pointer));
}

void fgets_test(){
    char buf[100];
    while (fgets(buf, 100, stdin) != NULL) {
        printf("-> %s", buf);
    }
}

void test_random() {
    int r = random();
    printf("exist random! %d\n", r);
}