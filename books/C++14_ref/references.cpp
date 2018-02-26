// At the same time as the reference is declared it must be 
// initialized with a variable of the specified type.
// 
// ref is an alias, like ref in Python
// so, when we need a ref, we should always use ref instead of pointer
// when the pointer need to reassigned, than we use pointer


#include <iostream>
using namespace std;

int main(){
    int x = 5;
    int &s = x;

    s = 10; 
    cout << "x = " << x << endl;
}
