#include <stdlib.h>
#include <stdio.h>

void exchange(int *a, int *b){
    int c = *a;
    *a = *b;
    *b = c;
}

int partition(int A[], int n, int p, int r){
    
    int x = A[r];                   /* pivot */
    int i = p-1;                    /* spilt index */
    for(int j=p; j<=r-1; j++){
        if (A[j] <= x) {
            i++;
            exchange(&A[i], &A[j]);
        }
    } /* end for */
    exchange(&A[i+1], &A[r]);
    return i+1;
}

void quickSort(int A[], int n, int p, int r){
    int q;
    if (p < r) {
        q = partition(A, n, p, r);
        quickSort(A, n, p, q-1);
        quickSort(A, n, q+1, r);
    }
}

void printSeq(int A[], int n){
    for (int i=0; i<n; i++)
        printf("%3d", A[i]);
    puts("");
}

int quickSelect(int A[], int n, int p, int r, int i){
    /* this function is go wrong */
    if (p == r)
        return A[p];
    /* the q-th element is already sorted! */
    int q = partition(A, n, p, r);
    /* the actual index of A[q] */
    int k = q - p + 1; 
    if (i == k)
        return A[q];
    else if (i<k)
        return quickSelect(A, n, p, q-1, i);
    else
        return quickSelect(A, n, p+1, r, i-k);
}    




int main(int argc, char* argv[]){
    int A[] = {2, 8, 7, 1, 3, 5, 6, 4};
    int n = 8;
    int p = 0;
    int r = 7;
    printSeq(A, n);
    int number = 8;
    int s = quickSelect(A, n, p, r, number);
    printf("the %dth element of A : %d \n", number, s);

    quickSort(A, n, p, r);
    printSeq(A, n);
    
    s = quickSelect(A, n, p, r, 8);
    printf("the %dth element of A : %d \n", 8, s);


    return 0;
}
