/* numeric literals prefix:
 * 0:  Octal literals
 * 0x: hexadecimal 
 * 0b: binary
 */


#include <iostream>
using namespace std;


int main(){
    int myOct = 032;
    int myHex = 0x32;
    int myBin = 0b0011'0010;

    cout << "myOct: " << myOct << endl
         << "myHex: " << myHex << endl
         << "myBin: " << myBin << endl;
}
