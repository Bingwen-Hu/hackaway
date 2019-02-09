#include <iostream>
#include <iterator>
#include <vector>

void modify(int& x)
{
    x = 10;
}

int triple(int x)
{
    return 3 * x;
}

// using const to prevent accidently modification
// rule of thumb is to keep the const keyword as close as 
// possible to whatever it is modifying
void print_vector(std::vector<int> const& v)
{
    // const test
    // v.front() = 42;
    std::cout << "{";
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>{std::cout, ""});
    std::cout << "}\n";
}

void add(std::vector<int>& v, int a)
{
    for (auto iter{v.begin()}, end{v.end()}; iter != end; ++iter){
        *iter = *iter + a;
    }
}


int main()
{
    int a{42};
    modify(a);
    std::cout << "a=" << a << std::endl;

    int b{triple(14)};
    std::cout << "b=" << b << std::endl;

    std::vector<int> data{10, 20, 40, 50};

    print_vector(data);
    add(data, 42);
    print_vector(data);
}
