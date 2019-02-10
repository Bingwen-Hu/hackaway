// type casting 
#include <iostream>

typedef signed char byte;

void print(byte value)
{
    // The << operator treats signed char as a mutant char, and tries to
    // print a character. In order to print the value as an integer, you
    // must cast it to an integer type.
    std::cout << "byte=" << static_cast<int>(value) << std::endl;
}

void print(short value)
{
    std::cout << "short=" << value << std::endl;
}


int main()
{
    print(static_cast<byte>(24));
    print(static_cast<short>(0));
}
