// tuple
#include <tuple>
#include <iostream>
using namespace std;

int main(){
    auto t = make_tuple(1, 2, 0.3, string("4"));
    cout << get<0>(t) << endl;
    get<2>(t) = 3.0;
    cout << get<double>(t) << endl;
    string s = get<3>(move(t));
}
