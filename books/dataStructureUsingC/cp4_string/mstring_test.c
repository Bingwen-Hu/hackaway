#include <stdio.h>
#include <stdlib.h>
#include "mstring.h"

int test_append() {
    /* test append */
    char *str1 = "Jenny meets ";
    char *str2 = "Mory";
    char *result;
    append(str1, str2, &result);
    printf("append string is: %s\n", result);
    free(result);
    return 0;
}

int main() {
    test_append();
}