// general utilities

#include <iostream>
#include <utility>

using namespace std;


void f(string s) {
    cout << "Moved or copied: " << s << endl;
}

void g(string&& s) {
    cout << "Moved " << s << endl;
}

string h(){
    string s("test");
    return s;
}

int main(){
    string test("123");
    f(test);
    f(move(test));

    test = "456";

    g(move(test));
    g(string("789"));
    g(h());
}
