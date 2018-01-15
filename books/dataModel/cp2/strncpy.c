#include <string.h>
#include <stdio.h>
#include <stdlib.h>


int main(){
    char *hello = malloc(sizeof(char) * 30);
    char hello2[] = "Mory";

    strncpy(hello, "Hi there.", 30);
    printf("%s\n", hello);
    strncpy(hello, hello2, 30); 
    printf("add hello2 %s\n", hello);

    // hack it!
    hello[4] = 'A';
    hello[5] = 'n';
    hello[6] = 'n';
    hello[7] = '\0';
    printf("After hack: %s\n", hello);

    // make hack easy
    strncat(hello, " Jenny", 30);
    printf("see who comes: %s\n", hello);

    free(hello);
}
