#include <stdio.h>

/* several way to define a matrix. */
void printMatrix(int matrix[][3], int row, int column);


void main(){

  
  int mat1[4][3];
  int mat2[4][3] = {{1, 2, 3}, {5, 6, 7}, {9, 10, 11}, {4, 8, 12}};
  
  int mat3[4][3] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  static int mat4[4][3];


  puts("init as mat[4][3]");
  printMatrix(mat1, 4, 3);
  
  puts("init as mat[4][3] = {{.3.}, {.3.}, {.3.}, {.3.}}");
  printMatrix(mat2, 4, 3);
  
  puts("init as mat[4][3] = {.12.}");
  printMatrix(mat3, 4, 3);

  puts("init as static mat[4][3]");
  printMatrix(mat4, 4, 3);
  
}


void printMatrix(int matrix[][3], int row, int column){

  for (int i=0; i<row; i++){
    for (int j=0; j<column; j++){
      printf("%d\t", matrix[i][j]);
    }
    putchar('\n');
  }
}

/* Test Note:

   only the first dimension can miss.
   and we can pass a matrix to a function as define.

 */
