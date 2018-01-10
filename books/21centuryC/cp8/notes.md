### notes about linkage
extern means different .c file will use the same one
static has two meanings, one is memory model, one is internal linkage


### Noun-adjective form
The trick to reading declarations is to read from right to left.

int const
    A constant integer

int const *
    A (variable) pointer to a constant integer

int * const
    A constant pointer to a (variable) integer

int * const *
    A pointer to a constant pointer to an integer

int const * * 
    A pointer to a pointer to a constant integer

int const * const *
    A pointer to a constant pointer to a constant integer

###### type and const can switch, but not const and *

