#include "fact.h"

long factorial(int n) {
    if (n < 0) {
        return 0l;
    } else if (n == 1 || n == 0) {
        return 1l;
    } else {
        return n * factorial(n-1);
    }
}

long factorial_tail(int n, long result) {
    if (n < 0) {
        return 0;
    } else if (n == 1 || n == 0) {
        return result;
    } else {
        return factorial_tail(n - 1, result * n);
    }
}