// quicksort in C++
#include <vector>
#include <iostream>

using namespace std;


void swap(vector<int>& vec, int i, int j) 
{
    int t = vec.at(i);
    vec[i] = vec.at(j);
    vec[j] = t;
}

int partition(vector<int>& vec, int start, int end) 
{
    int x = vec.at(end-1);
    int i = start;

    for (int j = start; j < end - 1; j++) {
        if (vec.at(j) < x) {
            swap(vec, j, i);
            i++;
        }
    }
    swap(vec, i, end-1);
    return i;
}

void quicksort(vector<int>& vec, int start, int end) 
{
    if (start < end) {
        int q = partition(vec, start, end);
        quicksort(vec, start, q);
        quicksort(vec, q+1, end);
    }
}

void print(vector<int>& vec)
{
    for (int i : vec) {
        cout << i << " ";
    }
    cout << endl;
}


int main()
{
    vector<int> vec{2, 8, 12, 7, 9, 14, 5, 6, 16, 1, 3, 10};
    quicksort(vec, 0, vec.size());
    print(vec);
}