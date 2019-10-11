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

void test_color()
{
    printf(RED "can this work?\n" RESET);
    printf(BLUE "blue never stop\n" RESET);
    printf(BLUE_BOLD "blue never stop\n" RESET);
    printf(BLUE_DIM "blue std never stop\n" RESET);
    printf(ESC_STD_RED "long macro\n"RESET);
    printf(ESC_STD_RED_WHITE "white background\n"RESET);
    printf(ESC_BOLD_RED_WHITE "bold red white background\n"RESET);
}

void test_printf_str()
{
    printf("\033" "[31mhaha\n" "You never know");
}

void test_macroprintf()
{
    myprintf(RED "haha%d\n" RESET, 1);
}