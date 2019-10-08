#pragma once


typedef struct conscell {
    void* data;
    struct conscell* next;
} conscell;

conscell* lpush(conscell* list, void* data);
conscell* lpop(conscell* list);
