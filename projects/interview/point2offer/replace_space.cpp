#include <iostream>
#include <string>


using namespace std;


string replaceSpace(string input) {
    int length = input.size(); 
    for (int i = 0; i < length; i++) {
        int c = input[i];
        if (c == ' '){
            input.append("  ");
        }
    }
    int p1 = length - 1;
    int p2 = input.size() - 1;
    while (p1 > 0 and p2 > p1) {
        if (input[p1] == ' ') {
            input[p2--] = '0';
            input[p2--] = '2';
            input[p2--] = '%';
        } else {
            input[p2--] = input[p1];
        }
        p1--;
    }
    return input;
}


int main()
{
    string input = "A B C Mory Jenny";
    string output = replaceSpace(input);

    cout << "input is " << input << endl;
    cout << "output is " << output << endl;
}

