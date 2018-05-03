#include <stdio.h>

int main() {
    int x = 10;
    char ch = 'A';
    void *gp;

    gp = &x;
    printf("generic pointer as an int %d\n", *(int *)gp);
    gp = &ch;
    printf("generic pointer as a char %c\n", *(char *)gp);
}