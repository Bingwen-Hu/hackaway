#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include "utilities.h"

int main(){
    char *base = malloc(30); 
    strncpy(base, "Thanks Guru!", 30);
    astrncat(&base, " I will work hard!");
    astrncat(&base, " And finally reach!");
    printf("%s\n", base);
    free(base);
}
