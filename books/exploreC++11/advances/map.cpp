#include <iostream>
#include <map>
#include <string>


int main()
{
    std::map<std::string, int> counts{};
    std::string word{};
    while (std::cin >> word) {
        ++counts[word];
    }

    for (auto element : counts) {
        std::cout << element.first << '\t' << element.second << std::endl;
    }

    // search in a map
    // std::map<std::string, int>::iterator the{counts.find("the")};
    auto the{counts.find("the")};
    if (the == counts.end()) {
        std::cout << "\"the\": not found\n";
    } else if (the->second == 1) {
        std::cout << "\"the\": occurs " << the->second << " time\n";
    } else {
        std::cout << "\"the\": occurs " << the->second << " times\n";
    }
}