// c++ 14 quick reference
#include <iostream>
using namespace std;

int main(){
    int16_t myInt16 = 0;
    int32_t myInt32 = 12;
    char myChar = 'M';

    std::cout << myInt16 << endl
              << myInt32 << endl
              << myChar  << endl;


    int myOct = 062;
    int myHex = 0x32;
    int myBin = 0b0011'0011;

    std::cout << myOct << endl
              << myHex << endl
              << myBin << endl;

    bool myBool = false;
    bool yourBool = true;

    std::cout << myBool   << endl
              << yourBool << endl;

    return 0;
}
