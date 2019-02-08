#include <algorithm>
#include <iostream>
#include <vector>

bool compareData(int x, int y)
{
    return x > y;
}

int main()
{
    std::vector<int> data{};
    int x{};

    while (std::cin >> x)
        data.push_back(x);

    std::sort(data.begin(), data.end(), compareData);

    for (std::vector<int>::size_type i{0}, end{data.size()}; i != end; ++i)
        std::cout << data.at(i) << " ";
    std::cout << std::endl;

    std::cout << "using iterator" << std::endl;
    for (std::vector<int>::iterator i{data.begin()}, end{data.end()}; i != end; i++)
        std::cout << *i << " ";
    std::cout << std::endl;
}
