%module vec // module name

// for vector
%include <std_vector.i>

// using template, for vector<int>
using std::vector;
namespace std{
    %template(vectori) vector<int>; 
}

// include header 
%{
    #include "vec.h" 
%}

// parse header
%include "vec.h" 