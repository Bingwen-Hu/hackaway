#include <iostream>
#include <string>

void print(int i)
{
    std::cout << i;
}

void print(std::string str)
{
    std::cout << str;
}

void print(int i, bool newline)
{
    std::cout << i;
    if (newline) {
        std::cout << std::endl;
    }
}

void print(std::string str, bool newline)
{
    std::cout << str;
    if (newline) {
        std::cout << std::endl;
    }
}


int main()
{
    auto str = "function overload";
    print(str);
    print(str, true);
    print(123);
    print(123, true);
}
