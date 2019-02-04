#include <iostream>
#include <vector>


int main()
{
    std::vector<int> data{};
    int x{};
    bool all_zero{true};

    while (std::cin >> x) {
        data.push_back(x);
    }

    for (auto d : data) {
        if (d != 0) {
            all_zero = false;
            break;
        }
    }

    std::cout << "All zero? " << std::boolalpha << all_zero << std::endl;
}
