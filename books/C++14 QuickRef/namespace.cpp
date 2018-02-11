#include <iostream>
using namespace std;

int main(){
    cout << "Hello world" << endl;
    cout << "what you say: " << cin.get() << endl;

    cout << "Please Say again: " << (char)cin.get() << endl;

}


// note that: cin.get() read one char one time, explained as integer.
