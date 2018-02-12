#pragma once

#include <string>

enum class SurpasserLevel {
    Nothing = 0, 
    Sense = 1,
    Control = 2,
    Cooperate = 3, 
    Surpass = 4
};

enum SurpasserType : unsigned int{
    PlainPeople = 0,
    WindSurpasser, 
    LandSurpasser, 
    WaterSurpasser, 
    FlameSurpasser, 
    LightningSurpasser, 
    LightSurpasser
};

class Surpasser
{
    public:
        Surpasser();
        Surpasser(const std::string name, SurpasserType type, SurpasserLevel level);
        ~Surpasser();
        void showStatus();
        void meditation(int howlong);

    private:
        std::string s_name;
        SurpasserType s_type; 
        SurpasserLevel s_level;
        
};
