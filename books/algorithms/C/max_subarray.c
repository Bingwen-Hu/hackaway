/*** find out the maximum sum of a continuous array

The thought is very clear but the algorithms just don't work.

    A : {-1, -3, -5, -4, 95, -5, -8, -9, -5}
                  2   3   4   5   6
*/


#include <stdio.h>
#include <limits.h>

void find_max_crossing_subarray(int A[], int low, int mid, int high,
                                int *max_left, int *max_right, int *cross_sum){
    int left_sum = INT_MIN;
    int sum = 0;
    for (int i=mid-1; i>=low; i--){   // [low, mid)
        sum += A[i];
        if (sum > left_sum){
            left_sum = sum;
            *max_left = i;
        }
    }

    int right_sum = INT_MIN;
    sum = 0;
    for (int j=mid; j<high; j++){    // [mid, high)
        sum += A[j];
        if (sum > right_sum){
            right_sum = sum;
            *max_right = j;
        }
    }

    *cross_sum = left_sum + right_sum;
}


// note that, low is accessible, high is unaccessible
void find_maximum_subarray(int A[], int low, int high,
                           int *max_low, int *max_high, int *max_sum){
    int  left_low,  left_high,  left_sum;
    int right_low, right_high, right_sum;
    int cross_low, cross_high, cross_sum;

    if (high == low + 1){
        *max_low  = low;
        *max_high = low;
        *max_sum  = A[low];
    } else {
        int mid = (low + high)/2;
        find_maximum_subarray(A, low, mid, &left_low,  &left_high,  &left_sum);
        find_maximum_subarray(A, mid, high, &right_low, &right_high, &right_sum);
        find_max_crossing_subarray(A, low, mid, high, &cross_low, &cross_high, &cross_sum);

        if (left_sum >= right_sum && left_sum >= cross_sum){
            *max_low  = left_low;
            *max_high = left_high;
            *max_sum  = left_sum;
        } else if (right_sum >= left_sum && right_sum >= cross_sum){
            *max_low  = right_low;
            *max_high = right_high;
            *max_sum  = right_sum;
        } else {
            *max_low  = cross_low;
            *max_high = cross_high;
            *max_sum  = cross_sum;
        }
    }
}

void test_find_max_crossing_subarray(){
    int A[] = {-1, -3, -5, -4, -5, -5, -8, -9, -5};
    int low = 1, mid = 5, high = 8, cross_low, cross_high, cross_sum;
    find_max_crossing_subarray(A, low, mid, high, &cross_low, &cross_high, &cross_sum);
    printf("the cross_low is: %d\nthe cross_high is: %d\nthe cross_sum is: %d\n", 
            cross_low, cross_high, cross_sum);
    // test pass
}

void main(){
    //         0  1   2   3  4   5   6   7   8  9  10  11  12  13
    int A[] = {1, 3, -5, -4, 3, -7, 20, 10, 7, 8, -5, -9, -13, 14};
    int len = sizeof(A)/sizeof(int);

    int max_left, max_right, cross_sum;
    find_maximum_subarray(A, 0, len, &max_left, &max_right, &cross_sum);
    printf("Max left is: %d\nMax right is: %d\nCross sum is %d\n",
           max_left, max_right, cross_sum);
}
