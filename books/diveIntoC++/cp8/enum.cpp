// enum in CPP

#include <iostream>
using namespace std;

enum RainbowColor {
    RC_RED, RC_ORANGE, RC_YELLOW, RC_GREEN, RC_BLUE, RC_INDIGO, RC_VIOLET
};


int main(){

    RainbowColor color = RC_RED;

    switch(color){
    case RC_RED:
        cout << "red";
        break;
    case RC_ORANGE:
        cout << "orange";
        break;
    default:
        cout << "other color";
        break;
    }


}
