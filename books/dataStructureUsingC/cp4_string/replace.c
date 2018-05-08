#include <stdio.h>
#include <string.h>

int main() {
    char str[200] = "learn from huger man!";
    char pattern[] = "huger";
    char replace[] = "more power";
    char new[200];
    int len = strlen(pattern);
    int new_i = 0;

    printf("origin str: %s\n", str);

    int i = 0;
    while (str[i] != '\0') {
        int j = 0, k = i;
        while (str[k] == pattern[j] && pattern[j] != '\0') {
            k++; j++;
        }
        int copy_loop = k;
        if (pattern[j] == '\0') {
            int replace_i = 0;
            while (replace[replace_i] != '\0') {
                new[new_i++] = replace[replace_i++];
            }
            i += len;
        }
        new[new_i] = str[copy_loop];
        i++; copy_loop++; new_i++;
    }
    new[new_i] = '\0';
    printf("new str is %s\n", new);
}