#include <stdio.h>

void display(int (*)[3]);

int main() {
    int mat[3][3] = {
        [0][0] = 1, [0][1] = 6, [0][2] = 3,
        [1][0] = 12, [1][1] = 16, [1][2] = 23,
        [2][0] = 121, [2][1] = 161, [2][2] = 231,
    };
    display(mat);
    
    return 0;
}

void display(int (*mat)[3]) {
    
    printf("\n the elements of the matrix are: ");
    for (int i = 0; i < 3; i++) {
        printf("\n");
        for (int j = 0; j < 3; j++) {
            printf("\t %d", *(*(mat + i) + j));
        }
    }
    puts("");
}