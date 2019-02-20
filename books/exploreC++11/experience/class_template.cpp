#include <iostream>

template <class T>
class point {
public:
    point(T x, T y) : x_{x}, y_{y} {}
    T x() const {return x_;}
    T y() const {return y_;}
    void print();
private:
    T x_, y_;
};

template <class T>
void point<T>::print()
{
    std::cout << "point(" << this->x_ << "," << this->y_ << ")\n";
}

int main()
{
    point<int> int_point{42, 1225};
    point<double> d_point{19.76, 20.19};
    int_point.print();
    d_point.print();
}
