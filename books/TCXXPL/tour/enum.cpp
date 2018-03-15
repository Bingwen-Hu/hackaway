/* enumeration in cpp */
#include <iostream>


enum class Traffic_light {green, yellow, red};

Traffic_light& operator++(Traffic_light& t)
{
    switch (t) {
        case Traffic_light::green: 
            return t = Traffic_light::yellow;
        case Traffic_light::yellow:
            return t = Traffic_light::red;
        case Traffic_light::red:
            return t = Traffic_light::green;
    }    
}

int main(){
    Traffic_light light = Traffic_light::red;
    ++light;

    switch (light) {
        case Traffic_light::red:
            std::cout << "red! Stop!" << std::endl;
            break;
        case Traffic_light::green:
            std::cout << "Let's go!" << std::endl;
            break;
        case Traffic_light::yellow:
            std::cout << "Take care!" << std::endl;
            break;
    }
}