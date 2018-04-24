#include "fact.h"
#include <stdio.h>

int main() {
    int n = 10;
    long f = factorial(n);

    printf("factorial of %d is %ld\n", n, f);


    int m = 12;
    long g = factorial_tail(m, 1);
    printf("factorial of %d is %ld\n", m, g);
}