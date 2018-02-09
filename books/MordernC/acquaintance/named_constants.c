// named constant page45

#include <stdio.h>

// rule 1.5.4.3 An object of const-qualified type is read-only
// rule 1.5.4.5 enumeration constants have either an explicit or positional value
// rule 1.5.4.6 Enumeration constants are of type `signed int`.
// rule 1.5.4.7 An integer constant expression doesn't evaluate any object.

int main(){

    enum corvid {magpie, raven, jay, chough, corvid_num, };

    char const* const animal2[corvid_num] = {
        [chough] = "chough",
        [raven] = "raven",
        [magpie] = "magpie",
        [jay] = "jay",
    };

    for (unsigned i = 0; i < corvid_num; ++i){
        printf("Corvid %d is the %s\n", i, animal2[i]);
    }

    char const* const animal[3] = {
        "raven",
        "magpie",
        "jay",
    };

    char const* const pronoun[3] = {
        "we",
        "you",
        "they",
    };

    char const* const ordinal[3] = {
        "first",
        "second",
        "third",
    };

    for (unsigned i = 0; i < 3; ++i){
        printf("Corvid %u is the %s\n", i, animal[i]);
    }

    for (unsigned i = 0; i < 3; ++i){
        printf("%s plural pronoun is %s\n", ordinal[i], pronoun[i]);
    }

    return 0;
}
