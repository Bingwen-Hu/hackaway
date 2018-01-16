#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>
#include "utilities.h"



int main(){
    //char *base = malloc(20);
    //strncpy(base, "mory is so good", 20);
    char *base = NULL;
    char *copyme = "Ann is better";
    astrncpy(&base, copyme);
    printf("%s\n", base);
    free(base);

}
