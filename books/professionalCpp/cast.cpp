/* three ways to cast a variable
 */

#include <iostream>
using std::cout;

int main(){
    float f = 3.14f;
    int i1 = (int)f;
    int i2 = int(f);
    int i3 = static_cast<int>(f);

    cout << i1 << i2 << i3 << "\n";
}
