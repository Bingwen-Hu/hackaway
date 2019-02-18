#include <algorithm>
#include <iostream>


#include "data.hpp"
#include "intrange.hpp"


int main()
{
    intvector data{};
    read_data(data);
    write_data(data);

    auto iter{std::find_if(data.begin(), data.end(), intrange{10, 20})};
    if (iter == data.end()){
        std::cout << "No values in [10, 20] found\n";
    } else {
        std::cout << "value " << *iter << " in range [10, 20].\n";
    }
}