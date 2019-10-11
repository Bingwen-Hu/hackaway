#include <stdio.h>
#include <stdlib.h>





void array_vs_pointer_in_str();
void fgets_test();
void test_random();
void test_sizeof();
void test_local_scope();
void test_valgrind();
void test_printf_str();

#define ESC "\033"
#define ESC_STD ESC"[0;"
#define ESC_STD_RED ESC_STD"31m"
#define ESC_STD_RED_WHITE ESC_STD"31;47m"
#define BOLD "[1"
#define RED ESC_STD_RED
#define BLUE ESC"[34m"
#define RESET ESC"[0m"

void test_macroexpand();