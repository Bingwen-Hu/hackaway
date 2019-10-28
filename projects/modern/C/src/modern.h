#pragma once
#include <stdio.h>

// Error
#define ferror(stream, msg) \
    fprintf((stream), "file %s, line %d: %s \n", __FILE__, __LINE__, (msg))

#define error(msg) \
    fprintf(stdout, "file %s, line %d: %s \n", __FILE__, __LINE__, (msg))