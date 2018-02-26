/* C++11 rvalue reference
 * rvalue ref bind and modify temporary objects or values.
 * allow avoiding unnecessary  copying and offer better performance
 */


#include <iostream>
using namespace std;

int main(){
    int &&rvalue = 1 + 3;

    rvalue += 4; // use as a normal variable

    cout << rvalue << endl;
}
