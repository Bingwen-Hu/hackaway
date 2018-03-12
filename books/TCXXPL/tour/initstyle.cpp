/* prefer {} initalization style */
#include <iostream>
#include <complex>

int main() 
{
    double d {2.3};
    std::complex<double> z2 {d, 2.4};

    int i2 {4};
    std::cout << z2.imag() << '\n';
    std::cout << i2 << '\n';
}