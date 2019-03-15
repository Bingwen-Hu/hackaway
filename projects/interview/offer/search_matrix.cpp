#include <iostream>

bool searchMatrix(int matrix[5][5], int target) 
{
    int i = 0, j = 4;
    while (true) {
        int search = matrix[i][j];
        if (target > search) {
            i++;
        } else if (target < search) {
            j--;
        } else {
            return true;
        }
        if (i > 4 or j < 0) {
            return false;
        } 
        
    }
    return false;
}

int main()
{
    int matrix[5][5] = {
        {1, 4, 7, 11, 15},
        {2, 5, 8, 12, 19},
        {3, 6, 9, 16, 22},
        {10, 13, 14, 17, 24},
        {18, 21, 23, 26, 30},
    };
    int target1 = 5;
    int target2 = 20;
    int target3 = 40;
    bool search1 = searchMatrix(matrix, target1);
    bool search2 = searchMatrix(matrix, target2);
    bool search3 = searchMatrix(matrix, target3);
    std::cout << "search for target1 " << search1 << std::endl;
    std::cout << "search for target2 " << search2 << std::endl;
    std::cout << "search for target3 " << search3 << std::endl;
}
