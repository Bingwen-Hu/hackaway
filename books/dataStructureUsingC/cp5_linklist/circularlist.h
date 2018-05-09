#ifndef CIRCULARLIST_H
#define CIRCULARLIST_H

typedef struct node {
    int data;
    struct node *next;
} node, *circularlist;

circularlist make();
circularlist insert_begin(circularlist list, int value);
circularlist insert_end(circularlist list, int value);
circularlist delete_begin(circularlist list);
circularlist delete_end(circularlist list);

void display(circularlist list);
void destroy(circularlist list);

#endif // CIRCULARLIST_H