/* array */

#include <iostream>
using namespace std;

int main(){

    int myArray[2][2] = {{0, 1}, {2, 4}};

    for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
            std::cout << myArray[i][j] << endl;


    /* dynamical array */
    int size = 4;
    int *d = new int[size];

    delete[] d;
}
