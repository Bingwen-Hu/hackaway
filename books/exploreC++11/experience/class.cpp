#include <iostream>

class work
{
public:
    work() = default;
    work(work const&) = default;
    work(std::string const& id, std::string const& title)
        : id_{id}, title_{title} 
    {}

    // why const before return body
    std::string const& id() const { return id_; }
    std::string const& title() const { return title_; }

private:
    std::string id_;
    std::string title_;
};
