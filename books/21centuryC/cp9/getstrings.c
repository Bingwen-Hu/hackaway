// compare to sadstring.c
#define _GNU_SOURCE // cause stdio.h to include asprintf
#include <stdio.h>
#include <stdlib.h>


void get_strings(char const *in){
    char *cmd;
    asprintf(&cmd, "strings %s", in);
    if (system(cmd)) fprintf(stderr, "something went wrong\n");
    free(cmd);
}

int main(int argc, char **argv){
    get_strings(argv[0]);
}
