/* head file for mory version string utilities */
#pragma once

void append(char *str1, char *str2, char **result);
void free_append(char *result);

void split(char *str, char sep, char ***result);
void free_split(char **strlist);

void replace(char *str, char *pattern, char **result);
void free_replace(char *result);

int equal(char *str1, char *str2);

void join(char **strlist, char *sep, char **result);
void free_join(char *result);

void substr(char *str, char *substr, int ***xys);
void free_substr(int **xys);

void strip(char *str, char **result);
void free_strip(char *result);