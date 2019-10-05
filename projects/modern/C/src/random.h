#pragma once

#include <stdlib.h>


inline int randint(int n) 
{
    return rand()/(RAND_MAX/n + 1);
}