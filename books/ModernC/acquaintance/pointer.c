// pointer
// Rule 1.6.2.6 Always initialize pointers
// Rule 1.6.3.1 Omitted struct initializers force the corresponding field to 0
// Rule 1.6.3.2 A struct initializer must initialize at least one field
// Rule 1.6.3.3 Struct parameters are passed by value

#include <stdio.h>

struct animalStruct {
    const char* jay;
    const char* magpie;
    const char* raven;
    const char* chough;
};

struct animalStruct const animal = {
    .chough = "chough",
    .raven = "raven",
    .magpie = "magpie",
    .jay = "jay",
};

