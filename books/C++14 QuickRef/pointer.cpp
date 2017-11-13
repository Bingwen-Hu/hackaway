// pointer

#include <iostream>
using namespace std;

int main(){

    int *d = new int;
    *d = 14;
    std::cout << "value of d is: "   << *d << endl
              << "address of d is: " << d  << endl;
    delete d;

    // null pointer type
    int *p = nullptr;  // ok
//    int i  = nullptr;  // error
    bool b = (bool)nullptr;  // ok

    std::cout << b << endl;

    nullptr_t mynull = nullptr;

}
