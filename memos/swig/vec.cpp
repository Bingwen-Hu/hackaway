#include "vec.h"
#include <vector>
#include <iostream>


void My::myvector(std::vector<int> const &v){
    for (int i : v) {
        std::cout << i << std::endl;
    }
}