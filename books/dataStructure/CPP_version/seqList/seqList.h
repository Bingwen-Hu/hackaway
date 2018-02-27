#pragma once

namespace Mory {

    class SeqList {
        private:
            static const int MAXSIZE = 100;
            int data[MAXSIZE];
            int length;

        public:
            SeqList();
            ~SeqList();
            int getElement(int index);
            int insertElement(int index, int value);
            int deleteElement(int index);
            int printElement();

    };
}