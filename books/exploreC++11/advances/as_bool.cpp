#include <iostream>
#include <vector>
#include <string>

void string_empty();
void vector_empty();

int main()
{
    if (true) std::cout << "true\n";
    if (false) std::cout << "false\n";
    if (42) std::cout << "42\n";
    if (0) std::cout << "0\n";
    if (42.4242) std::cout << "42.4242\n";
    if (0.0) std::cout << "0.0\n";
    if (-0.0) std::cout << "-0.0\n";
    if (-1) std::cout << "-1\n";
    if ('\0') std::cout << "'\\0'\n";
    if ('\1') std::cout << "'\\1'\n";
    if ("") std::cout << "\"\"\n";
    if ("1") std::cout << "\"1\"\n";
    if ("false") std::cout << "\"false\"\n";
    if (std::cout) std::cout << "std::cout\n";
    if (std::cin) std::cout << "std::cin\n";

    // string to bool test
    string_empty();
    vector_empty();
}

// Not all complex types can be converted to bool, however.

void string_empty()
{
    std::string empty{};
    if (empty)
        std::cout << "empty string is true\n";
    else
        std::cout << "empty string is false\n";
}

void vector_empty()
{
    std::vector<int> empty{};
    if (empty)
        std::cout << "empty vector is true\n";
    else
        std::cout << "empty vector is false\n";

}