#include <iostream>

template <class T>
class list
{
public:
    list():head_{nullptr}, tail_{nullptr}, size_{0} {}
    ~list() {clear();}
    void clear();
    void push_back(T const& x);
    void pop_back();
private:
    class node
    {
    public:
        node(T const& key): next_{nullptr}, prev_{nullptr}, value_{key} {}
        node* next_;
        node* prev_;
        T value_;
    };
    node* head_;
    node* tail_;
    std::size_t size_;
};

template <class T>
void list<T>::push_back(T const& x)
{
    node* n{new node{x}};
    if (tail_ == nullptr){
        head_ = tail_ = n;
    } else {
        n->prev_ = tail_;
        tail_->next_ = n;
        tail_ = n;
    }
    ++size_;
}

template <class T>
void list<T>::pop_back()
{
    node* n{tail_};
    if (head_ == tail_){
        head_ = tail_ = nullptr;
    } else {
        tail_ = tail_->prev_;
        tail_->next_ = nullptr;
    }
    --size_;
    delete n;
}

template <class T>
void list<T>::clear()
{
    for (std::size_t i=size_; i != 0; i--){
        pop_back();
    }
}

int main(){
    list<int> lst{};    
    lst.push_back(42);
    lst.push_back(42);
    lst.push_back(42);
    lst.push_back(42);
    lst.push_back(42);
    lst.push_back(42);
    return 0;
}
