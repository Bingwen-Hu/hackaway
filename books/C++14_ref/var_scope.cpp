// global variables are automatically initialized to zero
// while local variables may contain any garbage.

#include <iostream>

int global_var;

int main(){

    int local_var;
    std::cout << "global var:" << global_var << std::endl
              << "local var: " << local_var  << std::endl;
}
