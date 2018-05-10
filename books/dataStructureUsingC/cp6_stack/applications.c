/* application of stack */

#include "stack.h"
#include <stdio.h>

void reverse_list() {
    int list[] = {1, 2, 3, 4, 5, 7};
    int len = sizeof(list) / sizeof(int);

    printf("original list: ");
    for (int i = 0; i < len; i++) {
        printf("%3d", list[i]);
    }
    puts("");

    stack s = make();
    for (int i = 0; i < len; i++) {
        push(&s, list[i]);
    }
    for (int i = 0; i < len; i++) {
        list[i] = pop(&s);
    }
    printf("reverse list:  ");
    for (int i = 0; i < len; i++) {
        printf("%3d", list[i]);
    }
    puts("");
}

void parenthese_checker() {
    char *valid = "{ A + B * (C - D)}";
    char *invalid = "(A + B} * (C + I)";
    
    stack s = make();
    while (*valid != '\0') {
        
    }
}


int main(int argc, char const *argv[])
{
    reverse_list();
    return 0;
}
