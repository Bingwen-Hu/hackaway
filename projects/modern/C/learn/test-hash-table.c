#include <stdio.h>
#include "hash-table.h"

void test_strdup()
{
    char* s = "mory";
    char* dup = strdup(s);
    printf("duplicate string %s\n", dup);
    free(dup);
}

int main()
{
    // test_strdup();
    hash_table* ht = ht_new();
    ht_insert(ht, "name", "mory");
    ht_insert(ht, "gender", "man");
    ht_insert(ht, "teacher", "ann");
    ht_print(ht);
    ht_insert(ht, "name", "Jenny");
    ht_insert(ht, "gender", "woman");
    ht_print(ht);
    ht_delete(ht, "name");
    ht_print(ht);
    ht_free(ht);
}