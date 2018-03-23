#include "seqList.h"
#include <iostream>
using namespace Mory;
using namespace std;


int main(){
    
    SeqList list;
    list.insertElement(0, 1);
    list.insertElement(2, 2);
    list.insertElement(2, 3);

    list.printElement();
}