#include <iostream>

/**
 * In particular, absval imposes the following restrictions on T:
 * • T must be copyable. That means you must be able to copy an object of type T, so arguments can be passed to the function and a result can be returned. If T is a class type, the class must have an accessible copy constructor, that is, the copy constructor must not be private.
 * • T must be comparable with 0 using the < operator. You might overload the < operator, or the compiler can convert 0 to T, or T to an int.
 * • Unary operator- must be defined for an operand of type T. The result type must be T or something the compiler can convert automatically to T.
 */

template<class T>
T absval(T x)
{
    if (x < 0){
        return -x;
    } else {
        return x;
    }
}


int main()
{
    std::cout << absval(-42) << '\n';
    std::cout << absval(-4.2) << '\n';
    std::cout << absval(42) << '\n';
    std::cout << absval(4.2) << '\n';
    std::cout << absval(-42L) << '\n';
    std::cout << absval(-42) << '\n';
    std::cout << absval(-42) << '\n';
}
