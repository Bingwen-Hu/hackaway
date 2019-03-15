// welcome to cpp version!

#include <iostream>

class List
{
public:
    List(): head{new Node{0}} {}
    ~List() {
        while (head != nullptr) {
            auto p = head;
            head = head->next;
            delete p;
        }
    }
    int length() {return head->value;}
    void add(int value) {
        auto node{new Node{value}};
        node->next = head->next; 
        head->next = node;
        head->value++;
    }
    void print(){
        auto p = head->next;
        while (p != nullptr) {
            std::cout << p->value << " ";
            p = p->next;
        }
        std::cout << std::endl;
    }
private:
    struct Node
    {
        Node(int val) : value{val}, next{nullptr} {}
        int value; 
        Node* next;
    };
    Node* head;
};

int main()
{
    auto list = List(); 
    int input[7] = {20, 1, 34, 6, 8, 23, 10};
    for (int i=0; i < 7; i++) {
        std::cout << input[i] << " ";
        list.add(input[i]);
    }
    std::cout << std::endl;
    list.print();
}


