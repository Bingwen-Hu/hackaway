#include <stdlib.h>
#include "random.h"

inline int random(int n) 
{
    return rand()/(RAND_MAX/n + 1);
}