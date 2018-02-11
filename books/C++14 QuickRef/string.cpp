/* string
 * two way initialization (assign and direct-initialization)
 * handle with C-style string 
 * string compare
 * string length and modify
 * substr in C++, careful!
 */
#include <iostream>
#include <string>
using namespace std;


int main(){
   string h = "hello"; 
   string w(" world");
   string a = h + w;
   cout << a << endl;


   // C-style string
   char c[] = "world";
   string b = h + (string)c;
   string e = "hello" + (string)" world";

   cout << b << endl
        << e << endl;


   // String content compare
   // using == is ok
   string s = " world";
   cout << "s is equals to w? " << (s == w) << endl;

   // string length or size
   cout << "length of " << s << " is : " << s.length() << endl;
   cout << "s.substr(1, 5) is " << s.substr(1, 5) << endl;
}
