#include <iostream>
#include <map>

// using as a note
int main()
{
    typedef std::map<std::string,int> count_map;
    using count_iterator = count_map::iterator;

    count_map counts{};
    counts["Mory"] = 1;
    counts["Jenny"] = 2;
    count_iterator the{counts.find("Mory")};

    if (the == counts.end()) {
        std::cout << "No here!" << std::endl;
    } else {
        std::cout << "Found Mory!" << std::endl;
    }

}