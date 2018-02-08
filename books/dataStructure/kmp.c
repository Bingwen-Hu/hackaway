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

void compute_next(char *T, int *next, int len){

}


int main(){
    char str[] = "ababaaaba"; 
    int len = sizeof(str);
    int next[9] = {0};
    
    compute_next(str, next, len);

    for (int i=0; i<len; i++){
        printf("%2d", next[i]); 
    }
    puts("");
}
