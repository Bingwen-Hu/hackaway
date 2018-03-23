/* extra points
 * function property
 *
 */
#include <iostream>
using namespace std;


auto show_property() -> void 
{
    cout << "Enter function: " << __func__ << endl;
}

int main(){
    show_property();    
}
