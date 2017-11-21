#include <iostream>
#include <utility>
#include <ostream>

using namespace std;

class Person{;
public:
    Person() = default;
    explicit Person(const std::string& first,
                    const std::string& last = "",
                    bool isVIP = false)
        :m_first(first), m_last(last), m_isVIP(isVIP) {}

    const std::string& GetFirstName() const {return m_first;}
    void SetFirstName(const std::string& first) {m_first = first;}

    const std::string& GetLastName() const {return m_last;}
    void SetLastName(const std::string& last) {m_last = last;}

    bool IsVIP() const {return m_isVIP;}

private:
    friend bool operator<(const Person&, const Person&);
    std::string m_first;
    std::string m_last;
    bool m_isVIP = false;
};

bool operator<(const Person& lhs, const Person& rhs){
    if (lhs.IsVIP() != rhs.IsVIP()) return rhs.IsVIP();
    if (lhs.GetLastName() != rhs.GetLastName())
        return lhs.GetLastName() < rhs.GetLastName();
    return lhs.GetFirstName() < rhs.GetFirstName();
}

bool operator==(const Person& lhs, const Person& rhs){
    return lhs.IsVIP() == rhs.IsVIP() &&
        lhs.GetFirstName() == rhs.GetFirstName() &&
        lhs.GetLastName() == rhs.GetLastName();
}

std::ostream& operator<<(std::ostream& os, const Person& person){
    os << person.GetFirstName() << ' ' << person.GetLastName();
    return os;
}

int main(){

    std::pair<unsigned int, Person> p(42u, Person("Mory", "Demon"));
    // convenient
    auto p2 = std::make_pair(42u, Person("Ann", "Guevara"));

    // still need to improve.
    // cout << p << endl;
}
