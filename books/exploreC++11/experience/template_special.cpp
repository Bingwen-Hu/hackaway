#include <iostream>
#include <typeinfo>

// common template
template<class T>
class point
{
public:
    typedef T value_type;
    point(T const& x, T const& y) : x_{x}, y_{y} {}
    point() : point{T{}, T{}} {}
    T const& x() const {return x_;}
    T const& y() const {return y_;}
    void move_absolute(T const& x, T const& y){
        x_ = x;
        y_ = y;
    }
    void move_relative(T const& dx, T const& dy){
        x_ += dx;
        y_ += dy;
    }
    void typeinfo(){
        std::cout << "point<" << typeid(T).name() << ">()\n";
    }
private:
    T x_;
    T y_;
};


// special for int
template<>
class point<int>
{
public:
    typedef int value_type;
    point(int x, int y) : x_{x}, y_{y} {}
    point() : point{0, 0} {}
    int x() const {return x_;}
    int y() const {return y_;}
    void move_absolute(int x, int y){
        x_ = x;
        y_ = y;
    }
    void move_relative(int dx, int dy){
        x_ += dx;
        y_ += dy;
    }
    void typeinfo(){
        std::cout << "point<int> special\n";
    }
private:
    int x_;
    int y_;
};


int main()
{
    point<short> s;
    point<int> i;
    s.move_absolute(10, 20);
    i.move_absolute(42, 12);
    i.typeinfo();
    s.typeinfo();
}
