// string in cpp
#include <iostream>
#include <string>

using namespace std;

void get_name(){
    string first_name, last_name;

    cout << "Please enter your first name: ";
    cin >> first_name;
    cout << "Please enter your last name: ";
    cin >> last_name;
    string full_name = first_name + " " + last_name;
    cout << "Full name is: " << full_name;
}

int main(){
    string user_name;


//    cout << "Please enter your name: ";
//    cin >> user_name;
//    cout << "Hi " << user_name << "\n";
    get_name();
//    getline(cin, user_name, ',');
//    cout << user_name << endl;
//    getline(cin, user_name, '\n');
//    cout << user_name << endl;
    return 0;
}
