// quicksort in C++
#include <vector>
#include <iostream>

using namespace std;


int select_pivot(vector<int>& vec, int start, int end)
{
    // simple 
    return start;
}

int partition(vector<int>& vec, int start, int end) 
{
    q = select_pivot(vec, start, end);


    return q;
}

void quicksort(vector<int>& vec, int start, int end) 
{
    if (start < end) {
        q = partition(vec, start, end);
        quicksort(vec, start, q);
        quicksort(vec, q+1, end);
    }
}


int main()
{

}