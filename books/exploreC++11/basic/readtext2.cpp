#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

template<class C>
struct text : std::basic_string<C>
{
    text() : text{""} {}
    text(char const* s) : std::basic_string<C>(s) {}
    text(text&&) = default;
    text(text const&) = default;
    text& operator=(text const&) = default; 
    text& operator=(text&&) = default;
};


template<class Ch>
auto read(std::basic_istream<Ch>& in) -> std::vector<text<Ch>>
{
    std::vector<text<Ch>> result;
    text<Ch> line;
    while (std::getline(in, line))
        result.push_back(line);

    return result;
}


int main(int argc, char* argv[])
try
{
    std::cin.exceptions(std::ios_base::badbit);

    // part 1
    std::vector<text<char>> text;
    if (argc < 2)
        text = read(std::cin);
    else {
        std::ifstream in(argv[1]);
        if (not in) {
            std::perror(argv[1]);
            return EXIT_FAILURE;
        }
        text = read(in);
    }
    // part 2
    std::locale const& loc{ std::locale(argc >= 3 ? argv[2] : "") };
    std::collate<char> const& collate( std::use_facet<std::collate<char>>(loc) );
    std::sort(text.begin(), text.end(),
            [&collate](std::string const& a, std::string const& b)
            {
                return collate.compare(a.data(), a.data() + a.size(),
                                       b.data(), b.data() + b.size()) < 0; 
            }
        );
    // part 3
    for (auto const& line : text)
        std::cout << line << '\n';
} 
catch (std::exception& ex) 
{
    std::cerr << "Caught exception: " << ex.what() << '\n';
    std::cerr << "Terminating program.\n";
    std::exit(EXIT_FAILURE);
}
catch (...)
{
    std::cerr << "Caught unknown exception type.\n Terminating program.\n";
    std::exit(EXIT_FAILURE);
}
