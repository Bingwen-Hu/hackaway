#include "vec.h"
#include <vector>
#include <iostream>


void My::print_vector(std::vector<int> const &v){
    for (int i : v) {
        std::cout << i << std::endl;
    }
}