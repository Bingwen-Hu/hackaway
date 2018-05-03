#include <stdio.h>

int main() {
    printf("welcome to the world of C \f");
    printf("I am a positive number %+d\n", 42);
    printf("left align: as %-5d|\n", 42);
    printf("right align: as %5d|\n", 42);
    printf("hexadecimal without prefix: %0x\n", 42);
    printf("hexadecimal with prefix: %#0x\n", 42);
    printf("octal with prefix: %#o\n", 42);
    printf("octal without prefix: %o\n", 42);
    return 0;
}
