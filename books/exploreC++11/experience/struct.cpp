// The truth is quite simple. The struct and class keywords both start class definitions.  
// The only difference is the default access level: private for class and public for struct. That’s all
#include <iostream>
#include <sstream>
#include <cassert>

/// utility function
int gcd(int n, int m)
{
    n = std::abs(n);
    while (m != 0){
        int tmp{n % m};
        n = m;
        m = tmp;
    }
    return n;
}

/*********************************************************************
 * define struction with:
 * operator overload: != ==
 * member function: reduce assign print
 * access level: private public
 * declaration and definition: != ==
 * I/O customized: << >>
 *********************************************************************/
struct rational
{
    /// reduce the numerator and denominator by their GCD
    rational(int num, int den)
        : numerator{num}, denominator{den}
    {
        reduce();
    }

    rational()
        : rational{0, 1} ///< similar to  elixir
    {
        reduce();
    }

    // default constructor using `=`
    rational(rational const&) = default;
    
    /*
    rational& operator=(rational& rhs)
    {
        numerator = rhs.getNum();
        denominator = rhs.getDen();
        reduce();
        return *this;
    }
    */
    void reduce()
    {
        assert(denominator != 0);
        int div(gcd(numerator, denominator));
        numerator = numerator / div;
        denominator = denominator / div;
    }
    void print()
    {
        std::cout << numerator << "/"
                  << denominator << std::endl;
    }
    void assign(int num, int den)
    {
        numerator = num; 
        denominator = den;
        reduce();
    }

    int getNum();
    int getDen();
    

private:
    int numerator;
    int denominator{1};
};

int rational::getNum()
{
    return this->numerator; 
}

int rational::getDen()
{
    return this->denominator;
}

bool operator==(rational & a, rational & b)
{
    return a.getNum() == b.getNum() and a.getDen() == b.getDen();
}

inline bool operator!=(rational & a, rational & b)
{
    return not (a==b);
}

std::istream& operator>>(std::istream& in, rational& rat)
{
    int n{0}, d{0};
    char sep{'\0'};
    
    if (not (in >> n >> sep)){
        in.setstate(std::cin.failbit);
    } else if (sep != '/') {
        // a rational without a `/` so put it back
        in.unget();
        rat.assign(n, 1);
    } else if (in >> d){
        rat.assign(n, d);
    } else {
        in.setstate(std::cin.failbit);
    }
    return in;
}

std::ostream& operator<<(std::ostream& out, rational& rat)
{
    std::ostringstream tmp{};
    tmp << rat.getNum();
    if (rat.getDen() != 1){
        tmp << '/' << rat.getDen();
    }
    out << tmp.str();
    return out;
}


/*********************** End of definition ***********************/
/*****************************************************************/

int main()
{ 
    auto p1 = rational{12, 24};
    p1.reduce();
    rational p2{12, 20};
    std::cout << (p1==p2) << '\n';

    while (std::cin){
        if (std::cin >> p1){
            std::cout << p1 << '\n';
        }
    }
}
