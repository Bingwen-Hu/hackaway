#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

void read(std::istream& in, std::vector<std::string>& text)
{
    std::string line;
    while (std::getline(in, line))
        text.push_back(line);
}

int main(int argc, char* argv[])
{
    // part1 read standart input or a file
    std::vector<std::string> text;
    if (argc < 2)
        read(std::cin, text);
    else {
        std::ifstream in(argv[1]);
        if (not in){
            std::perror(argv[1]);
            return EXIT_FAILURE;
        }
        read(in, text);
    }
    
    // part2 sort the text
    std::sort(text.begin(), text.end());

    // part3 print the sorted text
    std::copy(text.begin(), text.end(),
            std::ostream_iterator<std::string>(std::cout, "\n"));

}

// mory note:
// compile with c++11 and using ctrl+D to exit, not ctrl+C
