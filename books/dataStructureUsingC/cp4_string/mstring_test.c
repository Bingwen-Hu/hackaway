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
    free_append(result);
    return 0;
}

int test_split() {
    char *str = "Jenny met mory ten years ago.";
    char **result;
    split(str, ' ', &result);

    char **p = result;
    while (*p != NULL) {
        printf("%s\n", *p++);
    }
    free_split(result);
    return 0;
}


int main() {
    test_append();
    test_split();
}