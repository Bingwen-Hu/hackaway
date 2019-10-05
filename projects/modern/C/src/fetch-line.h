#pragma once

#include <stdio.h>

char* fetch_line(char* buf, int buflen, FILE* stream, int* lineno);