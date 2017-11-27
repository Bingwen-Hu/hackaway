#include <stdio.h>

void println(int A[], int n){
    for (int i=0; i<n; i++){
        printf("%d ", A[i]);
    }
    putchar('\n');
}


void main(){
    int A[] = {9, 7, 5, 4, 2, 6, 18, 1, 0, 3};
    println(A, 10);
    bubbleSort2(A, 10);
    println(A, 10);
}
