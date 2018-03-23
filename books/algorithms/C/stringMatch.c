#include <stdio.h>
#include <string.h>


void naive_string_matcher(char *T, int length_T, char *P, int length_P){
    for (int i=0; i<length_T-length_P+1; i++){ // m-n+1
        int match = 1;
        for (int j=0; j<length_P; j++){        // m
            if (T[i+j] != P[j]){
                match = 0;
                break;
            } 
        } 
        if (match == 1){
            printf("Pattern occurs with shift %d\n", i); 
        }
    }
}

// test pass
void test_naive_string_matcher(){

    char T[] = "0001000101010001";
    char P[] = "0001";
    naive_string_matcher(T, strlen(T), P, strlen(P));

    char T2[] = "abcdadcdaedfdfcd";
    char P2[] = "cda";
    naive_string_matcher(T2, strlen(T2), P2, strlen(P2));
}

// KMP algorithms
void compute_prefix_function(char *P, int *next, int len){
}

void test_compute_prefix_function(){
    char P[] = "adaaadasad";
    int len = strlen(P);
    int next[10] = {0};
    compute_prefix_function(P, next, len);
    
    for (int i=0; i<len; i++){
        printf("%2d ", next[i]); 
    }
    puts("");
}

int main(){
}
