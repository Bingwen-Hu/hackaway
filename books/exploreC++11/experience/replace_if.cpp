#include <algorithm>

#include "data.hpp"
#include "intrange.hpp"



int main()
{
    intvector data{};
    read_data(data);
    write_data(data);
    std::cout << "using functor" << std::endl;
    std::replace_if(data.begin(), data.end(), intrange{10, 20}, 0);
    write_data(data);

    std::cout << "using lambda" << std::endl;
    std::replace_if(
        data.begin(), data.end(), 
        [](int x){return x == 0;},
        99);
    write_data(data);


}