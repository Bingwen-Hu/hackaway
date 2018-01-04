// macro with do while
#include <stdio.h>
#define doubleincrement(a, b) do {  \
    (a)++;                          \
    (b)++;                          \
} while(0)

int main(){
    int a = 2;
    int b = 44;

    if (a > b) doubleincrement(a, b);
    else return 0;

}
