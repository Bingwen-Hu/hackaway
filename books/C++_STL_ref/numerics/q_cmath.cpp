/* *
 * basic, using C library in C++
 * 
 * Note that: 
 *     fabs return an int
 */

#include <cmath>
#include <iostream>
using namespace std;

int main(){
    int x = 5;
    double y = -5.3;

    cout << "abs of " << x << " is " << abs(x)  <<endl;
    cout << "abs of " << y << " is " << fabs(x) <<endl;

    return 0;
}