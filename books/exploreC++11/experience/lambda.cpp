#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

// Lambdas are different from ordinary functions in that default for by-value 
// captures is const, and to get a non-const capture, you must explicitly tell the compiler. 
// The keyword to use is mutable, which you put after the function parameters,

void mutable_fn()
{
    int x{0};
    auto lambda = [&x](int y) mutable {
        x = 1;
        y = 2;
        return x + y;
    };

    int local{0};
    std::cout << lambda(local) << ", " << x << ", " << local << std::endl;
}


void capture()
{
    std::vector<int> data{};
    std::cout << "Multiplier: ";
    int multiplier{};
    std::cin >> multiplier;
    std::cout << "Data: \n";
    std::copy(std::istream_iterator<int>(std::cin),
              std::istream_iterator<int>(),
              std::back_inserter(data));

    // capture the local variable: multiplier
    std::transform(data.begin(), data.end(), data.begin(),
            [multiplier](int i){return i * multiplier;});

    std::copy(data.begin(), data.end(),
              std::ostream_iterator<int>(std::cout, "\n"));

}

int main()
{
    std::cout << "test mutable lambda" << std::endl;
    mutable_fn();

    std::cout << "capture local variable" << std::endl;
    capture();
}
