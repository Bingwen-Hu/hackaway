#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

// - A will contain between 1 and 21 elements, inclusive.
// - Each element of A will be between 0 and 220, inclusive.
std::string canBuild(int A[], int len) {
    int to_add{};
    for (int i{len-1}; i != 0; i--) {
        to_add = A[i] / 2;
        A[i-1] += to_add;
    }
    if (A[0] > 0) {
        return "Possible";
    } else {
        return "Impossible";
    }
}

void test_canBuild()
{
    int A[4]{0, 1, 2};
    std::string s1 = canBuild(A, 4);
    std::cout << s1 << std::endl;
    int B[2]{0, 3};
    std::string s2 = canBuild(B, 2);
    std::cout << s2 << std::endl;
    int C[5]{0,0,0,0,15};
    std::string s3 = canBuild(C, 2);
    std::cout << s3 << std::endl;
    int D[]{2,0,0,0,0,0,0,3,2,0,0,5,0,3,0,0,1,0,0,0,5};
    std::string s4 = canBuild(D, 21);
    std::cout << s4 << std::endl;
}

class ADPaper
{
public:
    ADPaper();
    ~ADPaper();
    string canBuild(vector<int> A);
    string canBuild(vector<int> A, int fast);
};

ADPaper::ADPaper(){};
ADPaper::~ADPaper(){};


string ADPaper::canBuild(vector<int> A){
    auto len{A.size()};
    int to_add{};
    for (auto i{len-1}; i != 0; i--) {
        to_add = A[i] / 2;
        A[i-1] += to_add;
    }
    if (A[0] > 0) {
        return "Possible";
    } else {
        return "Impossible";
    }
}

string ADPaper::canBuild(vector<int> A, int fast)
{
    long long compare = 1; 
    int len = A.size();
    long long x = 0;
    
    for (int i = 0; i < len; i++){
        x = A[i];
        if (x >= compare) {
            return "possible";
        }
        compare = compare << 1;
        if ((i+1) == len) {
            return "impossible";
        }
        A[i+1] += A[i] << 1;
    }
    return "impossible";
}


int main()
{
    // test_canBuild();
    auto adp{ADPaper()};
    vector<int> A{0,3};
    string s{adp.canBuild(A, 1)};
    cout << s << endl;
    vector<int> D{0,0,0,0,0,0,0,3,2,0,0,5,0,3,0,0,1,0,0,0,5};
    s = adp.canBuild(D, 1);
    cout << s << endl;
}
