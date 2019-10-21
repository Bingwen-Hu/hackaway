#include "vec.h"
#include <vector>

using std::vector;

int main(){
    My m = My();
    vector<int> v{};
    v.push_back(1);
    v.push_back(2);
    m.print_vector(v);
}