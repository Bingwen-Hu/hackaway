/* * sequence list in c++
 * 
 */

#include "seqList.h"
#include <iostream>

namespace Mory {
    SeqList::SeqList(): length(0) {};
    SeqList::~SeqList() {};

    int SeqList::getElement(int index) {
        return this->data[index];
    };
    int SeqList::insertElement(int index, int value) {
        while (index > this->length) {
            index--;
        }

        for (int i = index; i < this->length; i++) {
            this->data[i+1] = this->data[i];
        }
        this->data[index] = value;
        this->length++;
        return 0;
    };

    int SeqList::deleteElement(int index) {
        for (int i = index; i < this->length; i++) {
            this->data[i] = this->data[i+1];
        }
        this->length--;
        return 0;
    };
    int SeqList::printElement() {
        std::cout << this->length << " elements in seqlist:" << std::endl;
        
        for (int i = 0; i < this->length; i++){
            std::cout << i << " ";
        }
        std::cout << std::endl;
        return 0;
    };
}