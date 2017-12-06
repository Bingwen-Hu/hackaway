/** Class in an essay
1. define
2. member
3. method
4. inherit
5. access
*/


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

// method
void Surpasser::sense(){
    std::cout << "the " << this->surpassKind << " is flow...." << endl;
}

// constructor
Surpasser::Surpasser(){
    Surpasser("Surpasser", "nothing");
}
Surpasser::Surpasser(string name, string surpassKind) {
    this->name = name;
    this->surpassKind = surpassKind;

}


class WindSurpasser : public Surpasser{

}
WindSurpasser::WindSurpasser(string name, string surpassKind){
        this->name = name;
        this->surpassKind = surpassKind;
};


int main(){

    Surpasser mory("Mory", "thought");
    mory.sense();
    mory.waken();

    WindSurpasser NorthMaple("Lu", "Wind");
    NorthMaple.name = "Maple";
    NorthMaple.surpassKind = "Wind";
    NorthMaple.sense();
    NorthMaple.waken();
}
