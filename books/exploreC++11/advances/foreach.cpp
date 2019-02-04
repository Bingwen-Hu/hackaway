#include <iostream>
#include <iomanip>
#include <vector>
#include <iterator>

int main()
{
    std::vector<int> data{};    
    int x{};
    int width{6};

    // I mistakely add an ';' to while
    // make it strange behavior
    while (std::cin >> x) {
        data.push_back(x);
    }
    
    for (auto datum : data){
        std::cout << std::right 
                  << std::setw(width) << datum
                  << std::setw(width) << datum * 2
                  << std::setw(width) << datum * datum
                  << std::endl;
    }
}

