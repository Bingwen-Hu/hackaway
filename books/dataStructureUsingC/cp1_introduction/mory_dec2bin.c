/* mory exercise */

#include <stdio.h>

int input0_255(char *message) {
    int i, met = 0;
    do {
        printf("%s: ", message);
        scanf("%d", &i);
        if (0 <= i && i <= 255) {
            met = 1;
        } else {
            puts("Input valid! retry...");
        }
    } while (!met);

    return i;
}

int main() {
    int i = input0_255("Please input an int(0-255)");
    int mask = 128;
    int bit;
    while (mask > 0) {
        bit = (mask & i) > 0;
        printf("%d", bit);
        mask >>= 1;
    }
    puts("");
}