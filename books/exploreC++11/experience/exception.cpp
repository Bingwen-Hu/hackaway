#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>


int main()
{
    std::string line{};
    while (std::getline(std::cin, line)){
        try {
            line.at(10) = ' ';
            if (line.size() < 20){
                line.append(line.max_size(), '*'); 
            }
            for (std::string::size_type size(line.size());
                size < line.max_size();
                size = size * 2)
            {
                line.resize(size);
            }
            line.resize(line.max_size());
            std::cout << "okey\n";
        } catch (std::out_of_range const& ex){
            std::cout << ex.what() << '\n';
            std::cout << "string index (10) out of range.\n";
        } catch (std::length_error const& ex){
            std::cout << ex.what() << '\n';
            std::cout << "maximum string length (" << line.max_size() 
                      << ") exceeded.\n";  
        } catch (std::exception const& ex){
            std::cout << "other exception: " << ex.what() << "\n";
        } catch (...) {
            std::cout << "Unknown exception type. Program terminating.\n";
            std::abort();
        }
    }
}
