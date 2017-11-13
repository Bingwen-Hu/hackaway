#include <string>
#include <iostream>
using namespace std;


class Surpasser
{
public:
    string name;
    string surpassKind;


    Surpasser(); /* constructor */
    Surpasser(string name, string surpassKind);
    void sense();
    void waken(){std::cout << "I finally get to here..." << endl;};
};

void Surpasser::sense(){
    std::cout << "the " << this->surpassKind << " is flow...." << endl;
}

Surpasser::Surpasser(){
    Surpasser("Surpasser", "nothing");
}


Surpasser::Surpasser(string name, string surpassKind) {
    this->name = name;
    this->surpassKind = surpassKind;

}

int main(){

    Surpasser mory("Mory", "thought");
    mory.sense();
    mory.waken();

}
