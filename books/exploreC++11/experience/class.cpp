#include <iostream>
#include <iomanip>

class work
{
public:
    work() = default;
    work(work const&) = default;
    work(std::string const& id, std::string const& title)
        : id_{id}, title_{title} 
    {}

    virtual void info() const {
        using namespace std;
        cout << "ID: " << id() << endl;
        cout << "Title: " << title() << endl;
    };
    // why const before return body
    std::string const& id() const { return id_; }
    std::string const& title() const { return title_; }

private:
    std::string id_;
    std::string title_;
};


class book : public work
{
public:
    book() : work{}, author_{}, pubyear_{}
    {}
    book(book const&) = default;
    book(std::string const& id, std::string const& title, std::string const& author, int pubyear) 
        : work{id, title}, author_{author}, pubyear_{pubyear}
    {}
    std::string const& author() const { return author_; }
    int pubyear() const { return pubyear_; }
    void info() const override
    {
        using namespace std;
        cout << left;
        cout << setw(20) << "ID:" <<  this->id() << endl;
        cout << setw(20) << "Title:" << this->title() << endl;
        cout << setw(20) << "Author:" << this->author() << endl;
        cout << setw(20) << "Public Year:" << this->pubyear() << endl;
    }

private:
    std::string author_;
    int pubyear_;
};


int main()
{
    book b{"bookid", "title of book", "author of book", 2010}; 
    b.info();
    work w{"workId", "title of work}"};
    w.info();
}
