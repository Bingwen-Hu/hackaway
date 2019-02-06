// function style 

#include <iostream>
#include <string>
#include <vector>

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

int main()
{
    // test_canBuild();
    auto adp{ADPaper()};
    vector<int> A{0,3};
    string s{adp.canBuild(A)};
    cout << s << endl;
}