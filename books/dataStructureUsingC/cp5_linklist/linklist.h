#pragma once

typedef struct node {
    int data;
    struct node *next;
} node, *linklist;

linklist make();
void display(linklist list);
linklist insert_begin(linklist list, int value);
linklist insert_end(linklist list, int value);
void destroy(linklist list);
linklist sort(linklist list, int incr);