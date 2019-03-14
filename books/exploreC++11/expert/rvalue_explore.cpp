/**
 * The goal is to retain the benefit of creating objects on the stack, while eliminating unnecessary copying of data.
 */
#include <string>
#include <vector>
#include <iostream>

class mystring : public std::string
{
public:
    mystring() : std::string{} {std::cout << "mystring()\n"; }
    mystring(mystring const& copy) : std::string(copy) {
        std::cout << "mystring copy(\"" << *this << "\")\n";
    }
    // second version: add a move method to this class
    mystring(mystring&& move) noexcept
    : std::string(std::move(move)) {
        std::cout << "mystring move " << *this << std::endl;
    }
};


std::vector<mystring> read_data()
{
    std::vector<mystring> strings{};
    mystring line{};
    while (std::getline(std::cin, line)) {
        strings.push_back(std::move(line));
    }
    return strings;
}

int main()
{
    std::vector<mystring> strings{};
    strings = read_data();
}
