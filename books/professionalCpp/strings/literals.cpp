/* string literals
 * "hello world"  -> `const char*`
 * "hello world"s -> std::string
 *
 * Numeric Conversions
 * to_string
 *
 * String conversions
 * int                  stoi
 * long                 stol
 * unsigned long        stoul
 * long long            stoll
 * unsigned long long   stoull
 * float                stof
 * double               stod
 * long double          stold
 */
#include <iostream>
#include <string>

int main(){
    using std::cout;
    using std::string;

    const string s = "1234";
    cout << s << " string to long: " << stol(s) << '\n';

    float f = 12.344f; 
    cout << f << " float to string: " << std::to_string(f) << '\n';
}




