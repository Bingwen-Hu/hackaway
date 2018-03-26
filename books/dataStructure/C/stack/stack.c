/* using stack to calculate */
#include <stdio.h>

#define MAXSIZE 100
typedef struct stack {
    double data[MAXSIZE];
    int top; // top == len
} stack;

void push(stack *s, double value) {
    if (s->top < MAXSIZE) {
        s->data[s->top] = value;
        s->top++;
    }
}

double pop(stack *s) {
    if (s->top > 0) {
        s->top--;
        return s->data[s->top];
    }
    return -1.991234;
}

double calculate(stack *s, char op) {
    double op2 = pop(s);
    double op1 = pop(s);
    double result;
    switch (op) {
        case '+': result = op1 + op2; break;
        case '-': result = op1 - op2; break;
        case '*': result = op1 * op2; break;
        case '/': result = op1 / op2; break;
    }
    push(s, result);
    return result;   
}



int main() {
    stack s = {
        .data = {0},
        .top = 0,
    };

    char string[] = "9 3 1 - 3 * + 6 2 / +";
    double result;
    for (int i = 0; i < sizeof(string); i++) {
        if ('0' <= string[i] && string[i] <= '9') {
            push(&s, string[i] - '0');
        } else if ('+' == string[i] || '-' == string[i] || 
                   '*' == string[i] || '/' == string[i]) {
            result = calculate(&s, string[i]);
        }
    }
    printf("Result is %.2lf\n", result);
}