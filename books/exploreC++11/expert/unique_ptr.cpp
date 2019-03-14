/**
 * Keeping track of allocated memory can be tricky, so you should accept any help that C++ can offer. 
 * One class template that can help a lot is std::unique_ptr<> (defined in the <memory> header). 
 * This template wraps a pointer, so that when the unique_ptr object goes out of scope, it automatically deletes the pointer it wraps. 
 * The template also guarantees that exactly one unique_ptr object owns a particular pointer. Thus, when you assign one unique_ptr to another, you know exactly which unique_ptr (the target of the assignment) owns the pointer and has responsibility for freeing it. 
 * You can assign unique_ptr objects, pass them to functions, and return them from functions. In all cases, ownership passes from one unique_ptr object to another. Like children playing the game of Hot Potato, whoever is left holding the pointer or potato in the end is the loser and must delete the pointer.  
 *
 */

#include <iostream>
#include <memory>

class see_me
{
public:
    see_me(int x) : x_{x} {std::cout << "see_me " << x_ << std::endl;}
    ~see_me() {std::cout << "~see_me " << x_ << std::endl;} 
    int  value() const {return x_;}
private:
    int x_;
};

std::unique_ptr<see_me> nothing(std::unique_ptr<see_me>&& arg) 
{
    return std::move(arg);
}

template<class T>
std::unique_ptr<T> make(int x)
{
    return std::unique_ptr<T>{new T{x}};
}

int main()
{
    std::cout << "program begin... \n";
    std::unique_ptr<see_me> sm{make<see_me>(42)};
    std::unique_ptr<see_me> other;
    other = nothing(std::move(sm));
    
    if (sm.get() == nullptr) {
        std::cout << "sm is null, having lost ownership of its pointers\n";
    }
    if (other.get() != nullptr) {
        std::cout << "other now has ownership of the int, "
                  << other->value() << std::endl;
    }
    std::cout << "program ends..." << std::endl;
}



