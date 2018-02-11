/* std::array example
 *
 */


#include <iostream>
#include <array>
using namespace std;


int main(){
    array<int, 3> myArray = {0, 9, 1};
    cout << "Array size: " << myArray.size() << endl;
    cout << "Element 2 = " << myArray[1] << endl;
    return 0;
}
