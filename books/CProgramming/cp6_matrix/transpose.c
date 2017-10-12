/* Transpose a 2-d array */

#include <stdio.h>

#define ROW 3
#define COLUMN 4



void printMatrix(int matrix[][COLUMN], int row, int column);


void main(){

  int mat[ROW][COLUMN] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11};

  printMatrix(mat, ROW, COLUMN);
  
  
  int mat_t[COLUMN][ROW];
  
  for (int i=0; i<ROW; i++){
    for (int j=0; j<COLUMN; j++){
      mat_t[j][i] = mat[i][j];
    }
  }
  
  for (int i=0; i<COLUMN; i++){
    for (int j=0; j<ROW; j++){
      printf("%d\t", mat_t[i][j]);
    }
    putchar('\n');
  }


}


void printMatrix(int matrix[][COLUMN], int row, int column){

  for (int i=0; i<row; i++){
    for (int j=0; j<column; j++){
      printf("%d\t", matrix[i][j]);
    }
    putchar('\n');
  }
}




/* Test Note:

   how can I define a generic matrix print function?

 */
