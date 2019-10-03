#include <limits.h>
#include <stdio.h>
#include "leetcode.h"

int c_atoi(char *str) {
    char *p = str;
    int sign = 1;
    // skip the white space
    while (*p == ' ' && *p != '\0') p++;
    // judge the sign
    if (*p == '-') {
        sign = -1;
        p++;
    } else if (*p == '+') {
        p++;
    } else if (*p < '0' || *p > '9') {
        return 0;
    }
    // here we got very clean str!
    int ans = 0;
    while (*p != '\0' && *p >= '0' && *p <= '9') {
        int t = (*p - '0') * sign;
        if (ans > INT_MAX / 10 || (ans == INT_MAX / 10 && t > 7)) return INT_MAX;
        if (ans < INT_MIN / 10 || (ans == INT_MIN / 10 && t < -8)) return INT_MIN;
        ans = ans * 10 + t;
        p++;
    }
    return ans;
}

// https://leetcode-cn.com/problems/reverse-string/
void reverseString(char *s, int sSize) {

    // bounder check
    if (sSize == 0) { return; }

    char *head = s;
    char *tail = sSize + s - 1; // move point to last char
    char t = ' ';

    printf("head of string is %c\n", *head);
    printf("tail of string is %c\n", *tail);
    for (int i = 0; i < sSize / 2; i++) {
        t = *head;
        *head = *tail;
        *tail = t;    
        head++;
        tail--;
    }
}

