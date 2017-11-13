/* string */
#include <iostream>
#include <string>
using namespace std;

int main(){

    string h = "hello";
    std::cout << h << endl;


    string a = (string)"Hello" + "good"; // string cast is needed

    /* raw string is supported. */
    string cmd = R"(E:\Mory\gogs\haa)";  // () is needed
    std::cout << cmd <<endl;

    /* string function */
    size_t len = cmd.length();
    std::cout << "length of cmd is: " << len        << endl
              << "or using size(): "  << cmd.size() << endl;


    // string encode
    string  u3 = u8"utf-8 good";

    /* this two raise error

    u16string u16 = u"utf-16 oh";
    u32string u32 = U"Is it right";
    std::cout << u3  << endl
              << u16 << endl
              << u32 << endl;

    */
}
