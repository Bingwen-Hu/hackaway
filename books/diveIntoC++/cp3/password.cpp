// string compare
#include <iostream>
#include <string>

using namespace std;

int main(){

    string password;

    cout << "Enter your password" << "\n";
    getline(cin, password, '\n');

    if (password == "mory"){
        cout << "Access allowed\n";
    } else {
        cout << "Bad password. Denied!\n";
        return 0;
    }
    cout << "Enjoy!\n";
    return 0;
}
