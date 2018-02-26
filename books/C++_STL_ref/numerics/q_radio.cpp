#include <ratio>
#include <iostream>
#include <typeinfo>
using namespace std;

typedef std::ratio<1, 3> a_third;
typedef std::ratio<1, 2> a_half;
typedef std::ratio<2, 4> two_quart;
typedef std::ratio_add<a_third, a_half> sum;


int main(){

    cout << two_quart::num << '/' << two_quart::den << endl;
    cout << sum::num << '/' << sum::den << endl;
    cout << std::boolalpha;
    cout << (typeid(two_quart) == typeid(a_half)) << endl;
    cout << (typeid(two_quart::type) == typeid(a_half)) << endl;
    cout << std::ratio_equal<two_quart, a_half>::value << endl;

    return 0;
}



// test Note
// C++ error is very strange
