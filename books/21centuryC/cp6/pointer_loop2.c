// using typedef makes code clean
#include <stdio.h>
typedef char* string;

int main(){
    string list[] = {"First", "Second", "Third", NULL};
    for (string *p=list; *p != NULL; p++){
        printf("%s\n", *p); 
    }

}
