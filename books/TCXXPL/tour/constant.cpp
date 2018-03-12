/** constant in C++
 * const: value will not be changed
 * constexpr: evaluated at compile time
 */

#include <iostream>
int main()
{
    const int dmv = 15;                 // named constant
    constexpr double max1 = 1.4 * dmv;  // ok

    int var = 13;
    /* TODO: add try catch enable compile */
    // constexpr double max2 = 1.4 * var;  // error
}