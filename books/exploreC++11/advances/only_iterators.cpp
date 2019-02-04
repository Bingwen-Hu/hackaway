// Loops over iterator ranges are so common that many generic algorithms implement the most common actions that
// you may need to take in a program. With a couple of helpers, you can re-implement the program using only generic
// algorithms
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>


int main()
{
    std::vector<int> data{};    

    std::copy(std::istream_iterator<int>(std::cin),
              std::istream_iterator<int>(),
              std::back_inserter(data));
    // sort the vector
    std::sort(data.begin(), data.end());

    // print out
    std::copy(data.begin(), data.end(),
              std::ostream_iterator<int>(std::cout, " "));
}