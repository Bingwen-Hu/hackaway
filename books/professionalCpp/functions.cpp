/* extra points
 * function proper
 *
 */
#include <iostream>
using namespace std;


auto show_proper() -> void 
{
    cout << "Enter function: " << __func__ << endl;
}

int main(){
    show_proper();    
}
