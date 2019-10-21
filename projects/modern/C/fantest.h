#include <stdio.h>
#include <stdlib.h>


void array_vs_pointer_in_str();
void fgets_test();
void test_random();
void test_sizeof();
void test_local_scope();
void test_valgrind();
void test_printf_str();

#define myprintf(so, ...) printf(so __VA_ARGS__)
void test_macroprintf();

void test_strcpy();