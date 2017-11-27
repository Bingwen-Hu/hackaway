#include <stdio.h>

void merge(int A[], int p, int q, int r){
    const int n_left  = q - p;  // A[p] include, A[q-1] is largest
    const int n_right = r - q;  // A[q] include, A[r-1] is largest
    int Left[n_left];
    int Right[n_right];
    int i, j, k;

    //printf("n_left: %d\nn_right: %d\np: %d\nq: %d\nr: %d\n", n_left,
    //       n_right, p, q, r);
    for (i=0; i < n_left; i++){
        Left[i] = A[p+i];
        //printf("%d ", Left[i]);
    }
    for (j=0; j < n_right; j++){
        Right[j] = A[q+j];
        //printf("%d ", Left[j]);
    }

    i = j = k = 0;
    while (i < n_left && j < n_right){
        if (Left[i] <= Right[j]){
            A[p+k] = Left[i];
            //printf("A[%d]: %d\tLeft:%d\n", p+k, A[p+k], Left[i]);
            k++; i++;
        } else {
            A[p+k] = Right[j];
            //printf("A[%d]: %d\tRight:%d\n", p+k, A[p+k], Right[j]);
            k++; j++;
        }
    }


    while (i < n_left){
        A[p+k] = Left[i];
        //printf("A[%d]: %d\tLeft:%d\n", p+k, A[p+k], Left[i]);
        k++; i++;
    }
    while (j < n_right){
        A[p+k] = Right[j];
        //printf("A[%d]: %d\tRight:%d\n", p+k, A[p+k], Right[j]);
        k++; j++;
    }
}

mergeSort(int A[], int p, int r){
    if ((p+1) < r){
        int q = (p + r) / 2; // floor
        mergeSort(A, p, q);
        mergeSort(A, q, r);
        merge(A, p, q, r);
    }
    printf("A: ");
    for (int i = 0; i < 10; i++){
        printf("%d ", A[i]);
    }
    puts("");

}

void main(){
    int A[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    for (int i=0; i<10; i++){
        printf("%d ", A[i]);
    }
    puts("");
    //merge(A, 0, 5, 10);
    mergeSort(A, 0, 10);
    for (int i=0; i<10; i++){
        printf("%d ", A[i]);
    }
}
