#include <iostream>
#include "surpasser.h"

Surpasser::Surpasser(){};
Surpasser::Surpasser(const std::string & kind, int level){
    this->kind = kind;
    this->level = level;
}

Surpasser::~Surpasser(){};

void Surpasser::show(){
    std::cout << "Surpasser kind: " << this->kind << std::endl
              << "Surpassed level: " << this->level << std::endl;
}



