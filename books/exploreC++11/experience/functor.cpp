#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>


#include "functor.hpp"

int main()
{
    int size{};
    std::cout << "how many integers do you want? ";
    std::cin >> size;

    int first{};
    std::cout << "what is the first integer? ";
    std::cin >> first;

    int step{};
    std::cout << "what is the interval between successive integers? ";
    std::cin >> step;


    std::vector<int> data(size);
    std::generate(data.begin(), data.end(), sequence(first, step));

    std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));
}