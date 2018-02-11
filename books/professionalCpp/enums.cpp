#include <iostream>
using namespace std;

// weak type enum, config the type
enum SurpasserType : unsigned int{
    WindSurpasser = 0, 
    LandSurpasser, 
    WaterSurpasser, 
    FlameSurpasser, 
    LightningSurpasser, 
    LightSurpasser
};


// Strong type enum Note the class keyword
enum class SurpasserLevel {
    Nothing = 0, 
    Sense = 1,
    Control = 2,
    Cooperate = 3, 
    Surpass = 4
};

int main(){
    SurpasserType myType;
    myType = LandSurpasser;

    SurpasserLevel level = SurpasserLevel::Surpass;
    if (level == SurpasserLevel::Surpass){
        cout << "surpass level is highest!\n";  
    }

    switch (myType){
        case LightningSurpasser:
            cout << "Wow! Lightning surpasser!\n";
            break;
        case LightSurpasser:
            cout << "Oh my God! Light Surpasser!\n";
            break;
        default:
            cout << "Hello, what's your name?\n";  
            break;
    }

}
