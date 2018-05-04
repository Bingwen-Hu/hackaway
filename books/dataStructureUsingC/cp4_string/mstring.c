#include "mstring.h"
#include <string.h>
#include <stdlib.h>

void append(char *str1, char *str2, char **result) {
    int len1 = strlen(str1);
    int len2 = strlen(str2);

    *result = malloc(len1 + len2 + 1);
    for (int i = 0; i < len1; i++) {
        (*result)[i] = str1[i];
    }
    for (int i = 0; i < len2; i++) {
        (*result)[i + len1] = str2[i];
    }
    (*result)[len1+len2] = '\0';
}


void split(char *str, char sep, char **result) {
    char *p = str;
    int count = 1;
    while ((*p) != '\0'){
        if (*p == sep){
            count++;
        }
    } // end while

    int head, tail;
    head = tail = 0;
    for (int i = 0; i < count; i++) {
        
    }
}

