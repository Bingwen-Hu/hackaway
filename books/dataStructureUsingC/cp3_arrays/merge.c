/* merge two sorted array  */
#include <stdlib.h>
#include <stdio.h>

int *merge_array(int array1[], int array2[], int n1, int n2);

int main() {
    int array1[] = {1, 3, 5, 7, 9, 12, 32, 84};
    int array2[] = {42, 87, 1225, 1994};

    int *merged = merge_array(array1, array2, 8, 4);

    for (int i = 0; i < 12; i++) {
        printf("%d ", merged[i]);
    }
    puts("");

    free(merged);
}


int *merge_array(int array1[], int array2[], int n1, int n2) {

    int *merged = (int *)malloc(sizeof(int) * (n1 + n2));
    int k = 0, lp = 0, rp = 0;
    while (lp < n1 && rp < n2) {
        if (array1[lp] < array2[rp]) {
            *(merged + k) = array1[lp];
            lp++; k++;
        } else {
            *(merged + k) = array2[rp];
            rp++; k++;
        }
    }

    while (lp < n1) {
        *(merged + k) = array1[lp];
        lp++; k++;
    }
    while (rp < n2) {
        *(merged + k) = array2[rp];
        rp++; k++;
    }
    return merged;
}
