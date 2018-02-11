/* thing must note in auto
 * auto: only translate the core type of the initializer, which means that 
 * any reference and constant specifiers are dropped.
 * if wanted, dropped specifiers can be manually reapplied
 */


#include <iostream>
using namespace std;

int main(){
    int i = 10;
    int &ref = i;
    auto my = ref; // int, reference type is dropped
    cout << my << endl;

    auto& myref = ref; // reference type is applied
    cout << myref << endl;

    auto&& a = i; // int& lvalue ref
    auto&& b = 2; // int&& rvalue ref
    cout << a << " " << b << endl;

}
