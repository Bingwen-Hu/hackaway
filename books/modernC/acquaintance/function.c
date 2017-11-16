// function

// Rule 1.7.2.1 Use EXIT_SUCCESS or EXIT_FAILURE as return values of main.
// Rule 1.7.2.2 Reaching the end of the {} block of main is equlvalent to a return with value EXIT_SUCCESS
// Rule 1.7.2.3 Calling exit(s) is equivalent evaluation of return s in main'
// Rule 1.7.2.4 Exit never fails and never returns to its caller.


#include <stdio.h>
#include <stdlib.h>

size_t fibCacheRec(size_t n, size_t cache[n]){
    if (!cache[n-1]){
        cache[n-1] =
            fibCacheRec(n-1, cache) + fibCacheRec(n-2, cache);
    }
    return cache[n-1];
}


size_t fibCache(size_t n){
    if (n+1 <= 3)
        return 1;
    size_t cache[n];
    cache[0] =  1; cache[1] = 1;
    for (size_t i = 2; i < n; ++i)
        cache[i] = 0;
    return fibCacheRec(n, cache);
}


void fib2rec(size_t n, size_t buf[2]){
    if (n > 2){
        size_t res = buf[0] + buf[1];
        buf[1] = buf[0];
        buf[0] = res;
        fib2rec(n-1, buf);
    }
}

size_t fib2(size_t n){
    size_t res[2] = {1, 1,};
    fib2rec(n, res);
    return res[0];
}


void main(){
    size_t res = fibCache(10);
    printf("fibo of 10 is: %d\n", res);

    size_t res2 = fib2(10);
    printf("fibo of 10 is: %d\n", res2);
}
