#pragma once

typedef struct node {
    int data;
    struct node *next;
} node, *linklist;

linklist make();
void display(linklist list);

linklist insert_begin(linklist list, int value);
linklist insert_end(linklist list, int value);
linklist insert_at(linklist list, int index, int value);

linklist delete_begin(linklist list);
linklist delete_end(linklist list);
linklist delete_at(linklist list, int index);

void destroy(linklist list);
linklist sort(linklist list);
