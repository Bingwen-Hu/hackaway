#include <iostream>

int main()
{
    int *pointer{new int{42}};
    std::cout << *pointer << std::endl;
    *pointer = 10;
    std::cout << *pointer << std::endl;

    // delete
    delete pointer;
}
