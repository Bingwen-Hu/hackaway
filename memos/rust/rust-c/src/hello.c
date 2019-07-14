#include <limits.h> // get INT_MAX, INT_MIN
#include <stdio.h>

void hello() {
    printf("Hello, World!\n");
}

int double_it(int x) {
    return x * 2;
}

int myatoi(char *s) {
    int ans = 0;
    int start_parse = 0;
    int sign = 1;
    char *p = s;
    
    while (*p != '\0') {
        if (*p == ' ' && start_parse == 0) {
            p++;
            continue;
        } else if (*p == '-' && start_parse == 0) {
            start_parse = 1;
            sign = -1;
        } else if (*p == '+' && start_parse == 0) {
            start_parse = 1;
        } else if (*p >= '0' && *p <= '9') {
            // overflow check
            if (sign == 1 && (ans > INT_MAX / 10 || (ans == INT_MAX / 10 && *p >= '7'))) {
                return INT_MAX;
            } else if (sign == -1 && (-ans < INT_MIN / 10 || (-ans == INT_MIN / 10 && *p >= '8'))) {
                return INT_MIN;
            }
            start_parse = 1;
            ans = 10 * ans + (*p - '0');
        } else {
            break;
        }
        p++;
    }
    return ans * sign;
}