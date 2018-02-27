/* Class in C++
 * C++11 introduce default value in class
 * defalut access level is private
 * this keyword
 */
// #pragma once
#ifndef SURPASSER_H_
#define SURPASSER_H_

#include <string>

class Surpasser
{
private: 
    std::string kind = "Nothing";
    int level = 0;

public:
    Surpasser();
    Surpasser(const std::string & kind, int level);
    ~Surpasser();
    void show();
};


#endif
