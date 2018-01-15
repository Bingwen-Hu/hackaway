#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>

void astrncpy(char **base, char *copyme){
    int length = strlen(*base) + strlen(copyme) + 1; 
    printf("length is %d\n", length);
    *base = realloc(*base, length);
    strncpy(*base, copyme, length);
}


int main(){
    char *base = malloc(20);
    strncpy(base, "mory is so good", 20);
    char *copyme = " Ann is better";
    astrncpy(&base, copyme);
    printf("%s\n", base);

}
