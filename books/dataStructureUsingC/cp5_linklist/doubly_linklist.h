#pragma once

typedef struct node {
    int data;
    struct node *prev;
    struct node *next;
} node, *doublelinklist;

// doubly linked list with header
// lets try a different space.
doublelinklist insert_dl(doublelinklist list, int index, int value);
doublelinklist remove_dl(doublelinklist list, int index);
void destroy_dl(doublelinklist list);
void display_dl(doublelinklist list);
