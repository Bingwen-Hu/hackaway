#include <iostream>
#include <iomanip>

int main()
{
    using namespace std;

    // header
    cout << setw(4) << '*' << '|';
    for (int i{1}; i <= 10; i++)
        cout << setw(4) << i;
    cout << endl;

    // separator
    cout << setfill('-') << setw(4) << "" << '+'
         << setw(40) << "" << endl; 
    // restore the fill char
    cout << setfill(' ');
    // the matrix
    for (int i{1}; i <= 10; i++){
        // left of |
        cout << setw(4) << i << '|';
        // right of |
        for (int j{1}; j <= 10; j++){
            cout << setw(4) << i * j;
        }
        cout << endl;
    }
}