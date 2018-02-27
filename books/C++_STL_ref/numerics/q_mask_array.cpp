// mask_array
#include <valarray>
#include <iostream>

using namespace std;

int main(){

    valarray<int> ints = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10};
    valarray<bool> even = ((ints % 2) == 0);
    int count = std::count(begin(even), end(even), true);
    valarray<int> factors(4, count);
    ints[even] *= factors;

    for (auto &i : ints)
        cout << i << " ";
    cout << endl;

    // indirect_array
    valarray<int> ints_ = {0, 1, 2, 3, 4, 6};
    valarray<size_t> indices = {1, 2, 5};
    ints_[indices] = -100;
    for (auto &i : ints_)
        cout << i << " ";
    cout << endl;
}
