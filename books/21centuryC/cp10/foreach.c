#include <stdio.h>

#define foreach_string(iterator, ...)\
    for (char **iterator = (char*[]){__VA_ARGS__, NULL}; *iterator; iterator++)

typedef char* string;
#define foreach(iterator, ...)\
    for (string *iterator = (string[]){__VA_ARGS__, NULL}; *iterator; iterator++)
int main(){
    char *str = "thread";
    foreach_string(i, "yarn", str, "rope"){
        printf("%s\n", *i); 
    }
    foreach(i, "Good", "follow", "lots"){
        printf("%s\n", *i); 
    }
}
