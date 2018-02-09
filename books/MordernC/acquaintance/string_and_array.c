// string and array

// Rule 1.6.1.16 Using a string function with a non-string has undefined behavior

#include <stdio.h>
#include <string.h>


// string initialization
void sizeof_string(){
    char mory0[] = "mory";
    char mory1[] = {"mory"};
    char mory2[] = {'m', 'o', 'r', 'y', 0, };
    char mory3[5] = {'m', 'o', 'r', 'y', };

    // not string
    char mory4[4] = {'m', 'o', 'r', 'y'};

    printf("size of mory0: %u\n", sizeof(mory0));
    printf("size of mory1: %u\n", sizeof(mory1));
    printf("size of mory2: %u\n", sizeof(mory2));
    printf("size of mory3: %u\n", sizeof(mory3));
    printf("size of mory4: %u\n", sizeof(mory4));

}


// starts with "mem" are array functions
// while "str" are string functions

int main(int argc, char* argv[argc+1]){

    size_t const len = strlen(argv[0]);
    char name[len+1];

    memcpy(name, argv[0], len);
    name[len] = 0;
    if (!strcmp(name, argv[0])){
        printf("Program name \"%s\" successfully copied\n",
               name);
    } else {
        printf("Copying %s leads to different string %s\n",
               argv[0], name);
    }

    return 0;
}
