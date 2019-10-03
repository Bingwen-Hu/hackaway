#pragma once

#include <stdio.h>
#include <stdlib.h>

/* ======= Data Structure =======
Variable:
    ElemType: element type in list
Structure: 
    LinkList: list with head pointer
*/


typedef int ElemType;

typedef struct Node {
    ElemType data;
    struct Node *next;
} Node, *LinkList;
