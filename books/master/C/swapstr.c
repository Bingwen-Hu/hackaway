// reentrance

#include <stdio.h>
#include <string.h>

void swap(char p[], char q[]) {

}


int main() {
    char p[] = "mory";
    char q[] = "ann";

    printf("p -> %s, q -> %s\n", p, q);

    char c[100];
    strcpy(c, p);
    strcpy(p, q);
    // strcpy(q, c);
    printf("c -> %s\n", c);
    printf("p -> %s\n", p);
    printf("q -> %s\n", q);
}