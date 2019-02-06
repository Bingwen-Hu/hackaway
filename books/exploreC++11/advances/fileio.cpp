#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

void small_read()
{
    std::ifstream in{"foreach.cpp"};
    if (not in) {
        std::perror("foreach.cpp");
    } else {
        std::string x{};
        std::ofstream out{"fileio.out"};
        while (in >> x) {
            out << x << '\n';
            std::cout << x << '\n';
        in.close();
        out.close();
        }
    }
}


int main()
{
    small_read();
}