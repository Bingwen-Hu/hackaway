// WARNING: BUGGY
#include <stdio.h>
#include <stdlib.h>

typedef struct ListNode
{
    int value;
    struct ListNode* next;
} ListNode;

typedef struct List
{
    struct ListNode* head;
    int length;
} List;

void addNode(List* list, int value) {
    ListNode* node = (ListNode*)malloc(sizeof(ListNode));
    node->value = value;
    node->next = list->head->next;
    list->head->next = node;
    list->length++;
}

void printList(List* list){
    ListNode* p = list->head;
    while (p->next) {
        printf("%d ", p->value);
    } 
}

int main()
{ 
    int values[] = {10, 9, 23, 42, 19, 25};
    List list;   
    for (int i = 0; i < 6; i++)
    {
        addNode(&list, values[i]);
    }
    printList(&list);
}
