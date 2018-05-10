/* multistack, namely double stack 
   implement as array without error checking.
*/
#pragma once
#include <stdio.h>

#define STACK_SIZE 200

typedef struct mstack {
    int data[STACK_SIZE];
    int topA, topB;
} mstack;


void initstack(mstack *s);
int peekA(mstack *s);
int peekB(mstack *s);

int pushA(mstack *s, int value);
int pushB(mstack *s, int value);

int popB(mstack *s);
int popA(mstack *s);
