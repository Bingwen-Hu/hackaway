#include <vector>
#include <iostream>

using namespace std;



int preparePrize(vector<int> scores) {
    vector<int> prizes{1}; // the first person init as 1
    int num = scores.size();

    for (int i=1; i < num-1; i++) {
        if (scores[i] > scores[i-1]) {
            prizes[i] = prizes[i-1] + 1;
        } else if (scores[i] == scores[i-1]) {
            prizes[i] = 1;
        } else if (scores[i] < scores[i-1]) {
            prizes[i] = 1;
            // start adjust part
            if (prizes[i-1] == 1) {
                prizes[i-1] = 2;
                int p = i-1;    
                while (p > 0 and prizes[p] < prizes[p-1]) {
                    prizes[p-1] += 1; 
                    p--;
                }
            }
            // end adjust part
        }
        
    }
    // last position check
    // there are 9 conditions --- omit! :-)
    int last = num-1;
    if (scores[last] > scores[last-1] and scores[last] > scores[0]) {
        prizes[last] = scores[last-1] > scores[0] ? scores[last-1]+1 : scores[0]+1; 
    } else if (1){;}
}

int main(){
    return 0;
}
