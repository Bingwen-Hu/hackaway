#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "fetch-line.h"

static char* trim_line(char* s)
{
    // skip leading space
    while (isspace(*s)) {
        s++;
    }

    // ignore comments marked after '#'
    char* t = s;
    while (*t != '\0' && *t != '#') {
        t++;
    }
    if (*t == '#') {
        *t = '\0';
    }

    // ignore trailing space 
    t--;
    while (isspace(*t)) {
        *t = '\0';
        t--;
    }

    return s;
}

char* fetch_line(char* buf, int buflen, FILE* stream, int* lineno)
{
    char* s;
    if (fgets(buf, buflen, stream) == NULL) {
        return NULL;
    }

    *lineno += 1;
    if (buf[ strlen(buf) - 1 ] != '\n') {
        fprintf(stderr, "*** reading error: input line %d too "
                "long for %s's buf[%d]\n",
                *lineno, __func__, buflen);
        exit(EXIT_FAILURE);
    }

    s = trim_line(buf);
    if (*s != '\0') {
        return s;
    } else {
        return fetch_line(buf, buflen, stream, lineno);
    }
}