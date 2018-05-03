/* scanf recipe */
#include <stdio.h>

void basic() {
    int age;
    char name[20];
    scanf("%d,%s", &age, name);
    printf("My name is %s, I am %d years old\n", name, age);
}

void skip() {
    int age;
    char name[20];
    puts("input name as (age*name)");
    // when input, blank is ignore!
    scanf(" %d*%s", &age, name);
    printf("My name is %s, I am %d years old\n", name, age);
}

int main() {
    skip();
}