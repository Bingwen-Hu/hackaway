#include <iostream>
#include <vector>
#include <memory>


void revisit_unique_ptr()
{
    std::unique_ptr<int> ap{new int{42}};
    int *p{ap.release()};
    delete p;

    ap.reset(new int{10});
    ap.reset();
}


class see_me
{
public:
    see_me(int x) : x_{x} {std::cout << "see_me " << x_ << std::endl;}
    ~see_me() {std::cout << "~see_me " << x_ << std::endl;}
    int value() const {return x_;}
private:
    int x_;
};


std::shared_ptr<see_me> does_this_work(std::shared_ptr<see_me> x)
{
    std::shared_ptr<see_me> y{x};
    return y;
}

int main()
{
    revisit_unique_ptr();
    std::shared_ptr<see_me> a{}, b{};
    a = std::make_shared<see_me>(42);
    b = does_this_work(a);
    std::vector<std::shared_ptr<see_me>> v{};
    v.push_back(a);
    v.push_back(b);

    return 0;
}
