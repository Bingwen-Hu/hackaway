/* decltype C++11
 * similar to auto, but deduces the exact declared type, include ref
 * in C++14, auto may be used as expression for decltype.
 **/
#include <iostream>
using namespace std;


// deduce type: int
auto getValue(int x) {
    return x;
}
// deduce type: int&
decltype(auto) getRef(int& x){
    return x;
}


int main(){
    decltype(auto) b = 3; //int &&

    cout << b << endl;
}
