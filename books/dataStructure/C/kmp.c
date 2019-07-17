// 遗憾： 这个KMP估计只有我自己看得懂
// KMP
// 基本原理：KMP的目标是减少字符串匹配时的回溯次数，提高查找效率
// 如何做到呢？以下的博文提供了非常好的解答，也是本答案的来源。
// https://blog.csdn.net/qq_37969433/article/details/82947411

// KMP算法的难点在于计算NEXT数组。如果有了NEXT数据，那么在所匹配的字符串
// S中，如果第j位匹配失败，那么对j进行回溯，回溯的位置为：
// j - (j - next[j] + 1)
// 解释一下，减项 (j - next[j] + 1) 代表j应该向左回溯的位移
// j代表当前的位置，那么j的新位置，就是当前位置减去位移！
// 估计这个KMP只有我自己看得懂了

// 有一些资料并没有+1这个项，因为本例中的next数据是从索引的。而有些教程为了
// 计算的方便，就从1开始，原理上是一样的。

// TODO: a KMP.md is needed for future.
#include <stdio.h>
#include <stdlib.h>

void compute_next(char *P, int *next, int len){
    int i=0, j=0;
    next[0] = 0;

    while(i < len-1){ // 目前是求i+1位，so在len-1位求len位
        if (j == 0 || P[i] == P[j-1]){
            ++i; ++j; 
            next[i] = j;
            // printf("next[i] = next[%d] = %d\n", i-1, j);
        } else {
            j = next[j-1]; 
        }
    }
    puts("Finish compute next: ");
    for (int i = 0; i < len; i++) {
        printf("%d%c", next[i], i < len-1 ? ' ' : '\n');
    }
}

// https://leetcode-cn.com/problems/implement-strstr/solution/
int strStr(char * haystack, char * needle){
    int len = 0; 
    char *p = needle;

    // 计算needle的长度
    while (*p != '\0') { p++; len++; }
    if (len == 0) return 0;

    // 计算next数组
    int *next = (int*) malloc(sizeof(int) * len);
    compute_next(needle, next, len);

    // 匹配字符串
    // 基本思路，保持i不变，只对j进行回溯。
    int index = 0;
    int i = 0;
    int j = 0;
    while (haystack[i] != '\0') {
        while (j != len && haystack[i] != '\0') {
            // printf("compare h[i]=%c, n[i]%c \n", haystack[i], needle[j]);
            if (haystack[i] == needle[j]) {
                if (j == 0) index = i;
                i++; j++;
            } else if (j > 0) { 
                j = next[j] - 1; 
                index = i - j;
            } else {
                // j == 0
                i++;
            }
        }
        if (j == len) return index;
    }
    free(next);
    return -1;
}

void test_compute_next() {
    char str[] = "abcabcde"; 
    int len = 8;
    int next[8] = {0};
    
    compute_next(str, next, len);
}


void test_kmp() {
    char haystack1[] = "abcdefghij";
    char needle1[] = "ijk";
    int find1 = strStr(haystack1, needle1);
    printf("test 1:\n haystack=%s\n needle=%s\n find in position %d\n\n", 
        haystack1, needle1, find1
    );
    char haystack2[] = "python rust lisp c elixir c++";
    char needle2[] = "li";
    int find2 = strStr(haystack2, needle2);
    printf("test 2:\n haystack=%s\n needle=%s\n find in position %d\n\n", 
        haystack2, needle2, find2
    );
    char haystack3[] = "python rust lisp c elixir c++";
    char needle3[] = "";
    int find3 = strStr(haystack3, needle3);
    printf("test 3:\n haystack=%s\n needle='%s'\n find in position %d\n\n", 
        haystack3, needle3, find3
    );
    char haystack4[] = "mississippi";
    char needle4[] = "issip";
    int find4 = strStr(haystack4, needle4);
    printf("test 3:\n haystack=%s\n needle='%s'\n find in position %d\n\n", 
        haystack4, needle4, find4
    );
}

int main(){
    test_kmp();
}