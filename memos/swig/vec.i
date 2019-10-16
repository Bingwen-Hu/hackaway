%module vec

%include <std_vector.i>

using std::vector;

namespace std{
    %template(vectori) vector<int>;
}

%{
    #include "vec.h"
%}

%include "vec.h"
