#ifndef CIRCULARLIST_H
#define CIRCULARLIST_H

typedef struct node {
    int data;
    struct node *next;
} node, *circularlist;

circularlist make();
circularlist insert_begin(circularlist list, int value);
void display(circularlist list);
void destroy(circularlist list);

#endif // CIRCULARLIST_H