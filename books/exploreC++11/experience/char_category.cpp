#include <iomanip>
#include <iostream>
#include <map>
#include <string>


int main()
{
    typedef std::map<std::string, int> count_map;
    typedef std::string::size_type str_size;
    using string = std::string;

    count_map counts{};

    std::string okay{"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "abcdefghijlkmnopqrstuvwxyz"
                     "0123456789-_"};

    std::string word{};

    while (std::cin >> word) {
        // make a copy of word, keeping only the characters that
        // appear in okey
        std::string copy{};
        for (char ch : word){
            // using alias
            if (okay.find(ch) != string::npos) {
                copy.push_back(ch);
            }
        }
        if (not copy.empty()) {
            ++counts[copy];
        }
    }
    // determine the longest word
    str_size longest{0};
    for (auto pair : counts) {
        if (pair.first.size() > longest) {
            longest = pair.first.size();
        }
    }

    // for each word/count pair
    const int count_size{10};
    for (auto pair : counts) {
        std::cout << std::setw(longest) << std::left << pair.first
                  << std::setw(count_size) << std::right << pair.second
                  << std::endl;
    }
}
