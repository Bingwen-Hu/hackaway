#include "utilities.h"

void swap(int *first, int *second){
    int temp = *first;
    *first = *second;
    *second = temp;
}

void astrncpy(char **base, char *copyme){
     int length = strlen(copyme) + 1;
     *base = realloc(*base, length);
     strncpy(*base, copyme, length);
 }

void astrncat(char **base, char *addme){
    int length = strlen(*base) + strlen(addme) + 1;
    *base = realloc(*base, length);
    strncat(*base, addme, length);
}


void connect_mysql(char user[], char pass[], char db[]){
    apop_opts.verbose++;
    apop_opts.db_engine = 'm';
    strcpy(apop_opts.db_user, user);
    strcpy(apop_opts.db_pass, pass);
    apop_db_open(db);
}

