#pragma once


#define lfprint(stream, fmt, list, type)            \
    do {                                            \
        conscell* p;                                \
        for (p = (list); p != NULL; p = p->next) {  \
            type data = *(type *)(p->data);         \
            fprintf(stream, fmt, data);             \
        }                                           \
        fputc('\n', stream);                        \
    } while (0)


#define lprint(fmt, list, type) \
    lfprint(stdout, fmt, list, type)

typedef struct conscell {
    void* data;
    struct conscell* next;
} conscell;


conscell* lpush(conscell* list, void* data);
conscell* lpop(conscell* list);
