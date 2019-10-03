#include "xmalloc.h"

// generic function
#define make_vector(v, n)  \
    ((v) = xmalloc((n) * sizeof *(v)))