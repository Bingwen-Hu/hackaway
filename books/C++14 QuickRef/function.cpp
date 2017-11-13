/** function
function overload
auto and decltype
lambda expression
*/

#include <iostream>
#include <vector>
#include <functional>
using namespace std;

// functional pass
void call(int arg, function<void(int)> func){
    func(arg);
}

// default parameter
int myAdd(int a = 1, int b = 2){
    return a + b;
}

// function overload
void myFunction(string a, string b) { cout << a+" "+b; }
void myFunction(string a) { cout << a; }
void myFunction(int a) { cout << a; }

void autoShow(){
    vector<int> vec {1, 2, 3};
    for (auto& v: vec){
        std::cout << v << ' ';
    }

}


int main(){

    int c = myAdd();
    int d = myAdd(c);
    int e = myAdd(c, d);

    std::cout << c << endl
              << d << endl
              << e << endl;

    autoShow();

    // lambda expression
    auto printSquare = [](auto x){std::cout<<x*x;};
    call(2, printSquare);

    /// capture variables using &
    int a = 1;
    [&a](int x){a += x; }(2);
    std::cout << "a after capture by reference: " << a << endl;

    /// It is possible to specify a default capture mode, to indicate how any *unspecified*
    /// variable used inside the lambda is to be captured
    int b_ = 1, c_ = 2;
    [&, b_]() mutable {b_++; c_ += b_;}();
    std::cout << c_ << " " << b_ << endl;

    /// declare variable in capture scope
    int x = 1;
    [&, y = 2](){ x+=y;}();
    std::cout << x;

}
