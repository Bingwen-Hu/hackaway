/* lambda expression
 * [] lambda mark, used to capture freedom variable
 * []() parameters
 * -> type optional return type
 * {} lambda body
 */

#include <iostream>
using namespace std;

int main(){

    // python style
    auto sum = [](int x, int y) -> int {
        return x + y;
    };

    cout << sum(3, 4) << endl;

    // C++14 allow using auto in parameter
    auto sum14 = [](auto x, auto y) {return x + y; };
    cout << sum14(1.4, 3.9) << endl
         << sum14(3, 5) << endl;
}
