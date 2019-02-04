#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

int main()
{
    std::vector<int> data{};
    int x{};
    
    while (std::cin >> x) {
        data.push_back(x);
    }

    for (int d : data) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    for (auto p1{data.begin()}, p2{data.end()}; p1 != p2; /*empty*/) {
        --p2;
        // std::cout << "ending: " << *p2 << std::endl;
        if (p1 != p2) {
            // std::cout << "beginning: " << *p1 << std::endl;
            std::iter_swap(p1, p2);
            p1++;
        }
    }
    std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));

    // proven: easier to understand
    for (auto start{data.begin()}, end{data.end()}; 
        start != end and start != --end;
        ++start)
    {
        std::iter_swap(start, end);
    }
    std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));

    // or use reverse
    std::reverse(data.begin(), data.end());
    std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));

}