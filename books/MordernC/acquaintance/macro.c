// macro

// rule 1.5.4.10 Don't hide a terminating semicolon inside a macro
// rule 1.5.4.12 I is reserved for the imaginary unit.
#include <stdio.h>

enum corvid {magpie, raven, jay, chough, corvid_num, };
#define CORVID_NAME                 \
(char const* const[corvid_num]){    \
    [chough] = "chough",            \
    [raven] = "raven",              \
    [magpie] = "magpie",            \
    [jay] = "jay",                  \
}



void main(){

    for (unsigned i = 0; i < corvid_num; ++i)
        printf("Corvid %u is the %s\n", i, CORVID_NAME[i]);
}
