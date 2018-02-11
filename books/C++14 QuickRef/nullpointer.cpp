// c++11 introduced the keyword nullptr to distinguish between 0 
// and null pointer. nullptr can only be implicitly converted to 
// pointer and bool types.
//
// Note: delete an already deleted null pointer is safe.

#include <iostream>
using namespace std;

int main(){
    int *p = nullptr; // ok
    nullptr_t mynull = nullptr; // ok
    // bool b = mynull; error 
    bool b(mynull); // direct-initialized

    cout << p << " " << b << endl;
}
