/* Raw string C++11
 * like Python raw string
 */



#include <iostream>
#include <string>
using namespace std;

int main(){
    string escaped = "c:\\windows\\system32\\cmd.exe";
    string raw =  R"(c:\windows\system32\cmd.exe)";

    cout << escaped << '\n';
    cout << raw << '\n';

}
