/** new for loop */

#include <iostream>
using namespace std;
int main(){

    int a[3] = {1, 2, 3};

    for (int &i : a){
        std::cout << i << " ";
    }

    return 0;
}
