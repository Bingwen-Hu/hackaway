// only new style for loop introduces


#include <iostream>
using namespace std;


int main(){
    int a[3] = {1, 2, 3};
    for (int &i : a){
        cout << i; 
    }
    cout << endl;


    string s = "abc";
    for (char &c : s){
        cout << c; 
    }
    cout << endl;

}
