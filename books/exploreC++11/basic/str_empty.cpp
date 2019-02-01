#include <iostream>
#include <string>


int main(){
    std::string shape("Triangle");
    int sides;

    std::cout << "Shape\t\tSides\n" << 
                 "-----\t\t-----\n";
    std::cout << "Square\t\t" << 4 << '\n' <<
                 "Circle\t\t?\n";
    std::cout << shape << '\t' << sides << '\n';

    std::string emtpy;
    std::cout << "Empty\t\t" "|" << emtpy << '|' << std::endl;
}

// string object left without init is ok
// int will messy