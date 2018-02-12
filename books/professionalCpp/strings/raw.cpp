/* raw string with some carefulness.
 */


#include <iostream>
#include <string>
using namespace std;

int main()
{
    // basic 
    string str = R"(Hello "world" is ok)"; 
    cout << str << endl;

    // multiplelines
    string str2 = R"(I am a raw string
            Living with 2 lines\n\n)";
    cout << str2 << endl;

    // extend raw string
    string str3 = R"-(The character ) is embedded in this string)-";
    cout << str3 << endl;
}
