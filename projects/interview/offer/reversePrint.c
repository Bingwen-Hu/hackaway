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

void printList(List*);

List* makeList(){
    List* list = (List*)malloc(sizeof(List));
    list->head = (ListNode*)malloc(sizeof(ListNode));
    list->length = 0;
    list->head->next = NULL;
    return list;
}

void addNode(List* list, int value) {
    ListNode* node = (ListNode*)malloc(sizeof(ListNode));
    node->value = value;
    node->next = list->head->next;
    list->head->next = node;
    list->length++;
}

void printList(List* list){
    ListNode* p = list->head->next;
    while (p != NULL) {
        printf("%d ", p->value);
        p = p->next;
    } 
    printf("\n");
}

int main()
{ 
    int values[] = {10, 9, 23, 42, 19, 25};
    List* list = makeList();   
    for (int i = 0; i < 6; i++)
    {
        addNode(list, values[i]);
    }
    printList(list);
    return 0;
}
