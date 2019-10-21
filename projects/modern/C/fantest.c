#include <string.h>
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
    // switch to the following code to see the result
    // int r = random();
    int r = rand();
    printf("exist random! %d\n", r);
}

void test_sizeof() {
    int* r = malloc(sizeof(int));
    int size_p = sizeof(*r);
    printf("size of int is %ld\n", sizeof(int));
    printf("size of *r is %d\n", size_p);
}

void test_local_scope() 
{
    int i = 999;
    printf("outer i = %d\n", i);

    do {
        int i = 111;
        printf("inner i = %d\n", i);
    } while (0);
    {
        int i = 222;
        printf("Never confuse! %d\n", i);
    }
    printf("outer i = %d\n", i);
}

void test_valgrind()
{
    int *r = malloc(sizeof(int));
    *r = 12;
    printf("value of r is %d\n", *r);

    free(r);
}

void test_printf_str()
{
    printf("\033" "[31mhaha\n" "You never know");
}

void test_strcpy()
{
    char* s = "mory";
    char* a = malloc(sizeof(strlen(s)+1));
    char* c = strcpy(a, s);
    printf("value of a is %s\n", c);
    printf("len of s is %zu\n", strlen(s));
}