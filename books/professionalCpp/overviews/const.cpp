/* const usage: most frequent situations covers
 * 1. define const value (compared to #define xxx in C)
 * 2. protect parameters
 * 3. const references
 */
#include <iostream>

void mysteryFunction(const std::string* somestring){
    // usage 2
    // *something = "Text"; error!
}

void printString(const std::string& myString){
    // usage 3
    std::cout << myString << std::endl;
}

int main(){
    // usage 1
    const float versionNumber = 2.0f;
    const std::string productName = "Super Hyper Net Modulator";

    // usage 2
    std::string mystring = "Mory";
    mysteryFunction(&mystring);


    // usage 3
    printString(mystring);

}
