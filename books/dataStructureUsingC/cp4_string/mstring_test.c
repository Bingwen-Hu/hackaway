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

void test_equal() {
    char *p = "ann";
    char *q = "ann";

    char *r = "mory";

    int same = equal(p, q);
    printf("%s and %s is same? %s\n", p, q, same > 0 ? "Yes" : "No");
    same = equal(p, r);
    printf("%s and %s is same? %s\n", p, r, same > 0 ? "Yes" : "No");
}

int main() {
    test_append();
    test_split();
    test_equal();
}