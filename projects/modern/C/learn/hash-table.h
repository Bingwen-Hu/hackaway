#pragma once
#include <stdlib.h>
 
#define HASH_TABLE_SIZE 53
#define HASH_TABLE_PRIME_1 HASH_TABLE_SIZE
#define HASH_TABLE_PRIME_2 73

typedef struct ht_item {
    char* key;
    char* value;    
} ht_item;

typedef struct hash_table {
    size_t size;
    size_t count;
    ht_item** items; // a list of items pointer
} hash_table;


char* strdup(const char* s);

hash_table* ht_new();
void ht_free(hash_table* ht);

void ht_insert(hash_table* ht, const char* key, const char* value);
char* ht_search(hash_table* ht, const char* key);
void ht_delete(hash_table* ht, const char* key);
void ht_print(hash_table* ht);