// function objects
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
using namespace std;
template <typename T> struct my_plus {
    T operator() (const T& x, const T& y) const {return x+y;}
};


int main(){
    my_plus<int> functior;
    std::cout << functior(11, 22) << std::endl;


    // reference_wrapper
    int i = 234;
    std::vector<std::reference_wrapper<int>> v{std::ref(i)};
    v[0].get() = 432;
    std::cout << v[0] << "==" << i << std::endl;

    // predefine functor
    // `sort` in algorithm.h
    int array[] = {7, 9, 7, 2, 0, 4};
    sort(begin(array), end(array), std::greater<int>());

    // I want to see how the `begin` and `end` functions
    cout << begin(array) << " " << end(array) << endl;

    // same functor can used for different type.
    plus<> stl_functor;
    cout << functior(234, 432) << ' ' << functior(1.101, 2.045) << endl;

}

