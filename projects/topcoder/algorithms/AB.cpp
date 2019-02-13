#include <vector>
#include <string>
#include <iostream>


class AB
{
    string createString(int N, int K);
}

string AB::createString(int N, int K)
{
    int maxA = N / 2;
    int maxB = N - maxA;
    int residual = maxA * maxB - K;     

    if (residual < 0){
        return "";
    }


}

void test(AB& ab)
{
    
}

int main()
{

}