/* KMP string match algorithms
 * classic algorithms using Space exchange with Time 
 *
 * Key: build the NEXT array for the cursor to jump 
 * How: focus on the substring before the pointed element, couut from head
 * and tail to found out the longest substring, the NEXT value of that index
 * is len(substring)+1
 *
 * Example:
 * Give T = "ababaaaba"
 * j=0, point->a, str_before="",                            next[0] = 0
 * j=1, point->b, str_before="a",                           next[1] = 1
 * j=2, point->a, str_before="ab",                          next[2] = 1
 * j=3, point->b, str_before="aba",         pre=post="a",   next[3] = 2
 * j=4, point->a, str_before="abab",        pre=post="ab",  next[4] = 3
 * j=5, point->a, str_before="ababa",       pre=post"aba",  next[5] = 4
 * j=6, point->a, str_before="ababaa",      pre=post="a",   next[6] = 2
 * j=7, point->b, str_before="ababaaa",     pre=post="a",   next[7] = 2
 * j=8, point->a, str_before="ababaaab",    pre=post="ab"   next[8] = 3
 */

#include <stdio.h>

void compute_next(char *P, int *next, int len){
    // i 指向当前要计算next值的字符
    // j指向用来对比的字符
    int i=1, j=0;
    next[1] = 0;

    while(i <= len){
        if (j == 0 || P[i] == P[j]){
            ++i; ++j; 
            next[i] = j;
        } else {
            j = next[j]; 
        }
    }
}

// for test
void compute_single_next() {
    char str[] = "abaabcac"; 
    int len = 8;
    int next[9] = {0, 0, 1, 1, 2, 2, 3, 0, 0};
    int i = 7;

    printf("next[i] = %i\n", next[i]);
    int j = next[i - 1];
    printf("start compute from i-1\n\n");
    printf("value of next[i-1] = %i\n", j);
    while (1) {
        printf("%c compare with %c\n", str[i-1], str[j]);
        if (j == 0 || str[i-1] == str[j]) {
            j++;
            next[i] = j;
            break;
        } else {
            printf("next[j] change from %i to %i\n", j, next[j]);
            j = next[j];  
        }
    }
    printf("next[i] = %i\n", next[i]);
}

int main(){
    char str[] = "abaabcac"; 
    // index:   1  2  3  4  5  6  7  8
    // orginal: a  b  a  a  b  c  a  c
    // next be: 0, 1, 1, 2, 2, 3, 1, 2
    int len = sizeof(str);
    printf("len is %i\n", len);
    int next[9] = {0};
    
    compute_next(str, next, len);

    for (int i=1; i<len+1; i++){
        printf("%2d", next[i]); 
    }
    puts("");

    compute_single_next();
}
