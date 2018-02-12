// surpasser class definition
#include "surpasser.h"
#include <iostream>


namespace SurpasserLand {
    Surpasser::Surpasser(): 
        s_name("Who you are?"),
        s_type(PlainPeople),
        s_level(SurpasserLevel::Nothing){
        }

    Surpasser::Surpasser(const std::string name, SurpasserType type, SurpasserLevel level){
        s_name = name;
        s_type = type;
        s_level = level;
    }

    Surpasser::~Surpasser(){
        // nothing to do
    }
    void Surpasser::showStatus(){
        std::string type;
        switch (s_type){
            case WindSurpasser: 
                type = "WindSurpasser"; break;
            case FlameSurpasser:
                type = "FlameSurpasser"; break;
            case LandSurpasser:
                type = "LandSurpasser"; break;
            case WaterSurpasser:
                type = "WaterSurpasser"; break;
            case LightningSurpasser:
                type = "LightningSurpasser"; break;
            case LightSurpasser:
                type = "LightSurpasser"; break;
            default:
                type = "Plain People";
        }
        std::string level;
        switch (s_level){
            case SurpasserLevel::Sense:
                level = "basic sense"; break;
            case SurpasserLevel::Control:
                level = "intermediate control"; break;
            case SurpasserLevel::Cooperate:
                level = "high cooperate"; break;
            case SurpasserLevel::Surpass:
                level = "highest surpass!"; break;
            default:
                level = "No level at all, just a white paper.";
        }
        std::cout << "Surpasser Name: " << s_name << std::endl;
        std::cout << "Surpasser Type:  " << type  << std::endl
            << "Surpasser Level: " << level << std::endl;
    };

    void Surpasser::meditation(int howlong){
        std::cout << "Meditation " << howlong << " days to enlight ..." << std::endl;
    }
}
