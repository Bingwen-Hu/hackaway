#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "hash-table.h"

static ht_item HT_DELETED_ITEM = {NULL, NULL};

char* strdup(const char* s){
    size_t size = strlen(s) + 1;
    char* dup = malloc(size);
    strncpy(dup, s, size);
    return dup;
}




static ht_item* ht_item_new(const char* k, const char* v) 
{
    ht_item* i = malloc(sizeof(ht_item));
    i->key = strdup(k);
    i->value = strdup(v);
    return i;
}

hash_table* ht_new()
{
    hash_table* ht = malloc(sizeof(hash_table));
    ht->size = HASH_TABLE_SIZE;
    ht->count = 0;
    ht->items = calloc(ht->size, sizeof(ht_item));
    return ht;
} 

static void ht_item_free(ht_item* item)
{
    free(item->key);
    free(item->value);
    free(item);
}

void ht_free(hash_table* ht)
{
    for (int i = 0; i < ht->size; i ++) {
        ht_item* item = ht->items[i];
        if (item != NULL && item != &HT_DELETED_ITEM) {
            ht_item_free(item);
        }
    }
    free(ht->items);
    free(ht);
}



/**
 * @brief hash str to an integer limited by m
 * @param s, input string
 * @param a, prime number, should bigger than alphabet
 * @param m, hash table size
 * @return the index in hash table
 */
static size_t ht_hash(const char* s, const int a, const int m)
{
    long hash = 0;
    size_t len = strlen(s);
    for (int i = 0; i < len; i++) {
        hash += (long)pow(a, len - (i+1)) * s[i];
        hash = hash % m;
    }
    return (size_t)hash;
}

/**
 * @brief open address double hash to avoid collisions
 * @param s, input string
 * @param m, hash table size
 * @param attempt, number of times of collisions
 * @return the index in hash table
 */
static size_t ht_get_hash(const char* s, const int m, const int attempt)
{
    size_t hash_a = ht_hash(s, HASH_TABLE_PRIME_1, m);
    size_t hash_b = ht_hash(s, HASH_TABLE_PRIME_2, m);
    size_t hash = hash_a + attempt * (hash_b + 1) % m;
    return hash;
}


void ht_insert(hash_table* ht, const char* key, const char* value)
{
    ht_item* item = ht_item_new(key, value);
    size_t index = ht_get_hash(item->key, ht->size, 0);
    ht_item* cur_item = ht->items[index];

    int attempt = 1;
    // loop until finding NULL or DELETED position
    while (cur_item != NULL && cur_item != &HT_DELETED_ITEM) {
        // if duplicate key is found, update
        if (strcmp(cur_item->key, key) == 0) {
            ht_item_free(cur_item);
            ht->items[index] = item;
            return;
        }
        index = ht_get_hash(item->key, ht->size, attempt);
        cur_item = ht->items[index];
        attempt++;
    }
    ht->items[index] = item;
    ht->count++;
}

char* ht_search(hash_table* ht, const char* key)
{
    size_t index = ht_get_hash(key, ht->size, 0);
    ht_item* item = ht->items[index];

    int attempt = 1;
    // loop until find it or find it NULL
    while (item != NULL) {
        if (item != &HT_DELETED_ITEM) {
            if (strcmp(item->key, key) == 0) {
                return item->value;
            }
        }
       index = ht_get_hash(key, ht->size, attempt);
        item = ht->items[index];
        attempt++;
    }
    return NULL;
}

void ht_delete(hash_table* ht, const char* key)
{
    size_t index = ht_get_hash(key, ht->size, 0);
    ht_item* item = ht->items[index];

    int attempt = 1;
    while (item != NULL) {
        if (item != &HT_DELETED_ITEM) {
            if (strcmp(item->key, key) == 0) {
                ht_item_free(item);
                ht->items[index] = &HT_DELETED_ITEM;
            }
        }
        index = ht_get_hash(key, ht->size, attempt);
        item = ht->items[index];
        attempt++;
    }
    ht->count--;
}

void ht_print(hash_table* ht)
{
    for (int i = 0; i < ht->size; i++) {
        ht_item* item = ht->items[i];
        if (item != NULL && item != &HT_DELETED_ITEM) {
            printf("{%s: %s} ", item->key, item->value);
        }
    }
    printf("\n");
}