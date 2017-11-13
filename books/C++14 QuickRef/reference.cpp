/* reference is just an alias and has no its own space
   pointer is point to the address....
   when reassignment is not necessary, a reference is preferred.
*/



#include <iostream>
using namespace std;

int main(){

    /* right reference
       extends the lifetime of the temporary object and allows
       it to be used like an ordinary variable
    */

    int &&ref = 1 + 2;
    ref += 3;

    std::cout << ref << endl;


}
