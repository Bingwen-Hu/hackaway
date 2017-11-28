#include <stdio.h>

void println(int A[], int n){
    for (int i=0; i<n; i++){
        printf("%d ", A[i]);
    }
    putchar('\n');
}

void bubbleSort(int A[], int n){
    int t;
    for (int i=0; i<n; i++){
        for (int j=n-1; j>i; j--){
            if (A[j]<A[j-1]){
                t = A[j];
                A[j] = A[j-1];
                A[j-1] = t;
            }
        }

    }
}


void main(){
    int A[] = {9, 7, 5, 4, 2, 6, 128, 1, 10, 3};
    println(A, 10);
    bubbleSort(A, 10);
    println(A, 10);
}
