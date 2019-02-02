// C++ lets you align output fields to the right or the left. If you want to center a number, you are on your own.

#include <iomanip>
#include <iostream>

int main()
{
    using namespace std;
    cout << "|" << setfill('*') << setw(6) << 1234 << "|" << endl;
    cout << "|" << left << setw(6) << 1234 << "|" << endl;
    cout << "|" << setw(6) << -1234 << "|" << endl; // alignment is sticky
    cout << "|" << right << setw(6) << -1234 << "|" << endl;
}