#include <iostream>
#include <locale>
#include <map>
#include <string>


int main()
{
    using count_map = std::map<std::string, int>;
    // initialize stream
    // std::cout.imbue(std::locale{""});
    // but error
    // std::locale native{""};
    // std::cin.imbue(native);
    // std::cout.imbue(native);
    
    count_map counts{};

    std::string word{};

    while (std::cin >> word) {
        std::string copy{};
        for (char ch : word) {
            if (std::isalnum(ch, std::locale{})) {
                copy.push_back(tolower(ch));
            }
        }
        if (not copy.empty()) {
            ++counts[copy];
        }
    }
    for (auto pair : counts) {
        std::cout << pair.first << '\t' << pair.second << std::endl;
    }
}

