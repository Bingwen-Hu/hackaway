// operator
#include <iostream>
using namespace std;

int main(){

    // logic
    bool b = true && false;
    std::cout << b << endl;
    // bitwise operator

    int x = 5 & 4;  // and
        x = 5 | 4;  // or
        x = 5 ^ 4;  // xor
        x = 4 >> 1; // right shift
        x = 4 << 1; // left shift
        x = ~4;     // not


    std::cout << (false == 0) << endl;


    return 0;
}
