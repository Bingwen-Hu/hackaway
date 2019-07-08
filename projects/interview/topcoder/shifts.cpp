#include <vector>
#include <iostream>

using namespace std;



class Solution
{
public:
    string shiftingLetters(string S, vector<unsigned>& shifts) {
        int back_shift = 'z' - 'a' + 1;
        int shift_ = 0;

        transform(shifts);

        for (int i=0; i < shifts.size(); i++) {
            shift_ = shifts[i] % back_shift; 
            S[i] = S[i] + shift_;
            if (S[i] > 'z'){
                S[i] -= back_shift;
            } 
        }
        return S;
    }
    void transform(vector<unsigned>& shifts) {
        for (int i = shifts.size()-1 ; i > 0; i--) {
            // overflow
            shifts[i-1] += shifts[i];
        }
    }
};

int main()
{
    auto solution = Solution();
    vector<unsigned> shifts{3, 5, 9};
    auto s = solution.shiftingLetters("abc", shifts);
    cout << s << endl;
}