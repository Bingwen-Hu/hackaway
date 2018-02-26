/* lambda capture rule and example
 *
 */

#include <functional>
#include <iostream>
using namespace std;

// high order function
void call(function<void()> func) {func();}

/// basic_capture: a copy of a capture variable is used
void basic_capture(){
    int i = 2;
    auto printSquare = [i]() {cout << i*i << endl;};
    call(printSquare); 
}

/// reference capture
void reference_capture(){
    int a = 1;
    [&a](int x){ a += x;}(2); // define and call the function
    cout << a << endl;
}

/// capture mode
// [=] means captured by value 
// [&] means captured by reference
// Note: variables captured by value are normally constant,
// but the mutable specifier can be used to allow such variables
// to be modified.
void capture_mode(){
    int a = 1, b = 1;
    [=, &a]() mutable {b++; a += b;}();
    [&, b]() mutable {b++; a += b;} ();

    // a is changed, but b is not
    cout << a << " " << b << endl;
}

/// C++14, define variable in capture clause
void capture_defvar(){
    int a = 1;
    [&, b = 2]() { a += b; }();
    cout << a << endl;
    
    a = 1;
    [&, b = 2]() { a += b; }();
    cout << a << endl;
}


int main(){
    basic_capture();
    reference_capture();
    capture_mode();
    capture_defvar();
}
