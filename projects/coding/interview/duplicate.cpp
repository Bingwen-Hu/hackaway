#include <iostream>

#define N 30

bool anyDuplicate(int nums[], int length, int duplication[]) {
    for (int i=0; i < length; i++) {
        int n = nums[i];
        duplication[n]++; 
        if (duplication[n] > 1){
            return true;
        }
    }
    return false;
}

int main()
{
    int duplication[N]{};
    int nums[]{6, 2, 3, 5, 2, 0};
    int length = 6;
    int b = anyDuplicate(nums, length, duplication);
    std::cout << "duplicate? " << b << std::endl;
}
