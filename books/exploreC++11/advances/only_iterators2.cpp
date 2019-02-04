#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>


int main()
{
    std::vector<int> data{};

    data.insert(data.end(),
                std::istream_iterator<int>(std::cin), std::istream_iterator<int>());
    
    std::reverse(data.begin(), data.end());

    std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));
}