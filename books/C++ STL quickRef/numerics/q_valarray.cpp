// numeric arrays
#include <valarray>
#include <iostream>
using namespace std;


int main(){

    valarray<int> ints1(7);
    valarray<double> doubles = {1.1, 2.2, 3.3};
    int carray[] = {6, -5, 4, -3, 2, 1};
    valarray<int> ints2(carray, 6);
    valarray<int> ints3 = abs(ints2);

    for (auto &i: ints2)
        cout << i << " ";
    cout << endl;
    for (auto &i: ints3)
        cout << i << " ";
    cout << endl;

    // slice
    valarray<int> ints = {0, 1, 2, 3, 4, 5, 6, 7};
    slice mySlicer(2, 3, 2);
    const valarray<int>& constInts = ints;
    auto copies = constInts[mySlicer];
    auto refs = ints[mySlicer];
    valarray<int> factors{6, 3, 2};
    refs *= factors;

    cout << "as begin: ";
    for (int &i : ints)
        cout << i << " ";
    cout << endl;
}
