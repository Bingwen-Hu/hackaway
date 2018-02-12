/* one thing I don't know before is how to call a 
 * function which parameter is pass-by-reference
 */
#include <iostream>
using namespace std;

void addOne(int i){
    i++;
}

void addOne2(int& i){
    i++;
}

// rvalue to enable such call addOne3(4)
// but it seems unnecessary in this example
void addOne3(int&& i){
    i++;
}

int main(){
    int i = 8; 
    cout << "i originally equals " << i << endl;
    addOne(i);
    cout << "i after call pass-by-value: " << i << endl; 
    addOne2(i);
    cout << "i after call pass-by-ref: " << i << endl; 

    addOne3(4);
}
