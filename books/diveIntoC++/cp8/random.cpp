// random number

#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

int randRange(int low, int high){
    return rand() % (high - low) + low;
}


int main(){

    srand(time(NULL));
    for (int i=0; i<10; i++){
        cout << randRange(4, 10) << '\n';
    }
}
