#include <iostream>
#include <utility>

using namespace std;


struct A {A(){}; A(const A&) = delete;};
void f(const A&) {std::cout << "lval, ";}
void f(A&&)      {std::cout << "rval, ";}

// Three different forwarding (fwd)schemes;
template <typename T> void good_fwd(T&& t){
    f(std::forward<T>(t));
}
template <typename T> void bad_fwd(T&& t){
    f(t);
}
template <typename T> void ugly_fwd(T t){
    f(t);
}

int main(){
    A a;
    good_fwd(a); good_fwd(std::move(a)); good_fwd(A());
    bad_fwd(a);  bad_fwd(std::move(a));  bad_fwd(A());

}
