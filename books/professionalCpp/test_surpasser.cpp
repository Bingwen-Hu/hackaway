#include "surpasser.h"
#include <iostream>
#include <memory>
using namespace SurpasserLand;

int main(){
    // stack-based object
    Surpasser Mory;
    Mory.showStatus();
    std::cout << "====================" << std::endl;

    // heap-based object smart pointer used
    auto Ann = std::make_unique<Surpasser>("Ann", LightSurpasser, SurpasserLevel::Surpass);
    Ann->showStatus();
    std::cout << "====================" << std::endl;

    // heap-based object normal pointer used
    Surpasser* Jenny = new Surpasser("Jenny", WaterSurpasser, SurpasserLevel::Control);
    Jenny->showStatus();
    delete Jenny;
}
