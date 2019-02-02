#include <iostream>


int main()
{
    for (int i{0}; i != 10; i = i + 1)
        std::cout << i << '\n';
    // outside the scope, i is undefined
    // std::cout << i << std::endl;
    
    // listing 7-4 compute sum of integers from 10 to 20
    int sum{};
    for (int i{10}; i <20; i++)
        sum += i;
    std::cout << "sum from 10 to 20 is: " << sum << std::endl;
}