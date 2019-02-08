#include <iostream>
#include <vector>


int main()
{
    typedef std::vector<int> intvec;
    typedef intvec::iterator iterator;

    intvec xs, ys;

    {
        int x{}, y{};
        char sep{};
        while (std::cin >> x >> sep and sep == ',' and std::cin >> y){
            xs.push_back(x);
            ys.push_back(y);
        }
    }

    for (iterator x{xs.begin()}, y{ys.begin()}; x != xs.end(); ++x, ++y) {
        std::cout << *x << "," << *y << std::endl;
    }
}
