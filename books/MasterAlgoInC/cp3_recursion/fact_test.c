#include "fact.h"
#include <stdio.h>

int main() {
    int n = 10;
    long f = factorial(n);

    printf("factorial of %d is %ld", n , f);
}