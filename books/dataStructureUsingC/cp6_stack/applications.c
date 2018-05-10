/* application of stack */

#include "stack.h"
#include <stdio.h>
#include <string.h>

int eval_postfix(char *post);

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

int parenthese_checker(char *str) {
    
    stack s = make();
    while (*str != '\0') {
        if (*str == '(' || *str == '{') {
            push(&s, *str);
        }
        if (*str == ')') {
            if ('(' != pop(&s)) {
                return -1;
            }
        }
        if (*str == '}') {
            if ('{' != pop(&s)) {
                return -1;
            }
        }
        str++;
    }
    return 1;
}


void infix2postfix() {
    char *infix = "5 + 2 * 9 - 1 + 4 * (3 + 2) - 1 * 7";
    char *str = infix;
    char post[200];
    int i = 0;
    stack s = make();
    push(&s, '(');
    while (*str != '\0') {
        if (*str == '(') {
            push(&s, *str);
        }
        if ('0' <= *str && *str <= '9') {
            post[i++] = *str;
        }
        if (*str == ')') {
            char c;
            while ((c = pop(&s)) != '(') {
                post[i++] = c;
            }
        }
        if (*str == '*' || *str == '/') {
            char c = peek(&s);
            while (c == '*' || c == '/') {
                post[i++] = pop(&s);
                c = peek(&s);
            }
            push(&s, *str);
        }
        if (*str == '+' || *str == '-') {
            char c = peek(&s);
            while (c == '*' || c == '/' || c == '+' || c == '-') {
                post[i++] = pop(&s);
                c = peek(&s);
            }
            push(&s, *str);
        }
        str++;
    }
    while (s.top >= 0) {
        post[i++] = pop(&s);
    }
    post[i-1] = '\0';  // override the '('
    printf("infix: %s\n", infix);
    printf("postfix: %s\n", post);
    printf("value of postfix: %d\n", eval_postfix(post));
}

int eval_postfix(char *post) {
    stack s = make();
    char *p = post;
    int result;
    while (*p != '\0') {
        if ('0' <= *p && *p <= '9') {
            push(&s, *p - '0');
        } else {
            int op2 = pop(&s);
            int op1 = pop(&s);
            switch (*p) {
                case '+': result = op1 + op2; break;
                case '-': result = op1 - op2; break;
                case '*': result = op1 * op2; break;
                case '/': result = op1 / op2; break;
            }
            push(&s, result);
        }
        p++;
    }
    return result;
}

int main(int argc, char const *argv[])
{
    reverse_list();

    char *valid = "{ A + B * (C - D)}";
    char *invalid = "(A + B} * (C + I)";
    int good = parenthese_checker(valid);
    printf("%s is %s\n", valid, good > 0 ? "valid" : "invalid");
    int bad = parenthese_checker(invalid);
    printf("%s is %s\n", invalid, bad > 0 ? "valid" : "invalid");


    infix2postfix();
    return 0;
}
