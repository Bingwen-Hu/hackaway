#include <vector>
#include <string>
#include <iostream>

using namespace std;

class AB
{
public:
    string createString(int N, int K);
};

string AB::createString(int N, int K)
{
    int A = N / 2;
    int B = N - A;
    int residual = A * B - K;     

    if (residual < 0){
        return "";
    }

    string&& str = "", strA, strB;
    // exchange A and B totally
    int B_to_start = residual / A;
    str.assign(B_to_start, 'B');
    strA.assign(A, 'A');
    str.append(strA);
    strB.assign(B-B_to_start, 'B');
    str.append(strB);

    // move forward certain steps
    int B_to_step = B_to_start + A - residual % A;
    str[B_to_step] = 'B';
    str[B_to_start+A] = 'A';
    return str;
}

void test()
{
    AB ab{};
    string str = ab.createString(10, 12);
    cout << str << endl;
    str = ab.createString(5, 8);
    cout << str << endl;
    str = ab.createString(2, 0);
    cout << str << endl;
}

int main()
{
    test();
}