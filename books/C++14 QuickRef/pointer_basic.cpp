// dynamic allocation


#include <iostream>
using namespace std;


int main(){
    int *d = new int;

    *d = 12;

    cout << "address of " << *d << " is " << d << endl;
    
    delete d;
    
}
