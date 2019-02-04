// compare string and vector
#include <iostream>
#include <vector>
#include <string>

void compare_vector()
{
    std::string str1{"Abc"}, str2{"aB"};
    std::cout << "Abc > aB? " << std::boolalpha << (str1 > str2) << std::endl;
}

void compare_string()
{
    std::vector<int> vec1{42, 100}, vec2{12, 30, 201};
    // could not print!
    // std::cout << "vec1 = " << vec1 << std::endl;
    // std::cout << "vec2 = " << vec2 << std::endl;
    std::cout << "vec1 < vec2? " << std::boolalpha << (vec1 < vec2) << std::endl;
}

int main()
{
    compare_string();
    compare_vector();
}