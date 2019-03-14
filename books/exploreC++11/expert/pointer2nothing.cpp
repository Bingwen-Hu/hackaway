// C++ offers some tools to help you: one is a special pointer value that you can assign to any pointer variable to represent a “pointer to nothing.” 
// You cannot dereference a pointer to nothing, but you can copy and assign these pointers, and most important, compare a pointer variable with the “pointer to nothing” value. 
// In other words, when your program deletes an object, it should assign a “pointer to nothing” value to the pointer variable. 
// By ensuring that every pointer variable stores a valid pointer or a “pointer to nothing,” you can safely test whether a pointer is valid before dereferencing it.

#include <iostream>
#include <new>

int main(){

    // init: the two statements are equivalent
    int* ptr{};
    int* ptr2{nullptr};
    
    // you can assign a null pointer to a variable after you delete it, 
    // as a way to mark the pointer object as no longer pointing to anything.
    // A good programming practice is to ensure that no pointer variable ever retains an invalid value. 
    // Assigning nullptr to a pointer variable is one way to accomplish this
    int* ptr3{new int{42}};
    delete ptr3;
    ptr3 = nullptr;
    
    if (ptr3 == nullptr) {
        std::cout << "I catch a nullptr!" << std::endl; 
    }

    // you can ask that the new expression return a null pointer instead of //
    // throwing an exception when it cannot allocate enough memory for the new object. 
    // Just add (std::nothrow) after the new keyword and be sure to check the pointer that new returns. 
    // The parentheses are required, and std::nothrow is declared in <new>
    int *ptr4(new (std::nothrow) int{42});
    if (ptr4 != nullptr) {
        std::cout << "allocated successfully!" << std::endl;
    }
    delete ptr4;

}
