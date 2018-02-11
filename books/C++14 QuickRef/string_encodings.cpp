/* string encodings
 */

#include <iostream>
#include <string>
using namespace std;


int main(){
    string s3 = u8"An asterisk: \u002A";
    u16string s4 = u"utf-16 string";
    u32string s5 = U"utf-32 string";

    cout << s3 << endl;
}
