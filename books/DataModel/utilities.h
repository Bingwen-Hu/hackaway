#pragma once
#include <malloc.h>
#include <string.h>
#include <apop.h>

void swap(int *first, int *second);
void astrncpy(char **base, char *copyme);
void astrncat(char **base, char *addme);


// mysql
void connect_mysql(char user[], char pass[], char db[]);
