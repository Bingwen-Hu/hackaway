#include <iostream>

int main()
{
    std::cout << "true=" << true << std::endl;
    std::cout << "false=" << false << std::endl;
    // a manipulator flag for boolean
    std::cout << std::boolalpha;
    std::cout << "true=" << true << std::endl;
    std::cout << "false=" << false << std::endl;
}