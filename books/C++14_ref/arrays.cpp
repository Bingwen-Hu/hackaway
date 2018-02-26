/* arrays:
 * declaration and allocation
 * assginment
 * multi-dimensional arrays
 * dynamic arrays
 */

#include <iostream>
using namespace std;

void printArray(int array[], int size){
    for (int i = 0; i < size; i++){
        cout << array[i] << "  "; 
    }
    cout << endl;
}


int main(){
    // one way
    int myArray[3];
    myArray[0] = 1;
    myArray[1] = 2;
    myArray[2] = 3;
    printArray(myArray, 3);

    // second way
    // int my2[3] = {1, 2, 3};
    // int my3 = {1, 2, 3};
    
    // multi-dimensional arrays
    // int my4[2][2] = {{0, 1}, {2, 3}};
    int my5[2][2] = {0, 1, 2, 3};
    if (my5[0][1] == 1){
        cout << "C++ is rows first\n"; 
    } else {
        cout << "C++ is columns first\n"; 
    }
    
    // dynamic arrays
    int *p = new int[3];
    *(p+1) = 10;
    *(p) = 2;
    *(p+2) = 1;
    printArray(p, 3);
    delete p;
}
