/** Quick Sort
select the last as pivot

    0  1  2  3  4  5  6  7  8  9 10 11
--------------------------------------
 0: 2  8 12  7  9 14  5  6 16  1  3 10
 1: 2  8 12  7  9 14  5  6 16  1  3 10 (swap(A, 0,  0))
 2: 2  8 12  7  9 14  5  6 16  1  3 10 (swap(A, 1,  1))
 3: 2  8 12  7  9 14  5  6 16  1  3 10 (j = 2,    skip)
 4: 2  8  7 12  9 14  5  6 16  1  3 10 (swap(A, 2,  3))
 5: 2  8  7  9 12 14  5  6 16  1  3 10 (swap(A, 3,  4))
 6: 2  8  7  9 12 14  5  6 16  1  3 10 (j = 5,    skip)
 7: 2  8  7  9  5 14 12  6 16  1  3 10 (swap(A, 4,  6))
 8: 2  8  7  9  5  6 12 14 16  1  3 10 (swap(A, 5,  7))
 9: 2  8  7  9  5  6 12 14 16  1  3 10 (j = 8,    skip)
10: 2  8  7  9  5  6  1 14 16 12  3 10 (swap(A, 6,  9))
11: 2  8  7  9  5  6  1  3 16 12 14 10 (swap(A, 7, 10))
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void swap(int A[], int i, int j){
    int t = A[i];
    A[i] = A[j];
    A[j] = t;
}

void quicksort(int A[], int p, int r){
    if (p < r){
        int q = partition(A, p, r);
        quicksort(A, p, q);             // [p,   q-1)
        quicksort(A, q+1, r);           // [q+1, r-1)
    }
}

int partition(int A[], int p, int r){
    int x = A[r-1];
    int i = p;

    for (int j = p; j < r - 1; j++){
        if (A[j] <= x){
            swap(A, i, j);
            i++;
        }
        println(A, r);
    }
    swap(A, i, r-1);
    return i;
}

void println(int A[], int n){
    for (int i=0; i<n; i++){
        printf("%2d ", A[i]);
    }
    putchar('\n');
}

int randomized_partition(int A[], int p, int r){
    srand((unsigned)time(NULL));
    int i = rand() % (r-p) + p;                     // [p,   r)
    swap(A, i, r-1);
    int q = partition(A, p, r);
    return q;
}


void randomized_quicksort(int A[], int p, int r){
    if (p < r){
        int q = randomized_partition(A, p, r);
        randomized_quicksort(A, p, q);             // [p,   q-1)
        randomized_quicksort(A, q+1, r);           // [q+1, r-1)
    }
}


void main(){
    int A[] = {2, 8, 12, 7, 9, 14, 5, 6, 16, 1, 3, 10};
    int len = sizeof(A)/sizeof(int);
    println(A, len);
    //quicksort(A, 0, len);
    randomized_quicksort(A, 0, len);
    println(A, len);
}
