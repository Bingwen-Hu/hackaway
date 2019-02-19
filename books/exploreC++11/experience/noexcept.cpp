#include <iostream>
#include <exception>


// To tell the compiler that a function does not throw an exception, add the noexcept qualifier after the function parameters (after const but before override).
void function() noexcept
{
    throw std::exception{};
}

int main()
{
    try {
        function();
    } catch (std::exception const& ex) {
        std::cout << "Gotcha!\n";
    }
}


