// two ways to generate random
// True non-deterministic random number std::random_device

#include <random>
#include <iostream>
#include <functional>
#include <vector>
using namespace std;


int main(){

    default_random_engine generator;
    uniform_int_distribution<int> distribution(1, 6);
    int dice_roll = distribution(generator);
    cout << "Random number: " << dice_roll << endl;

    function<int()> roller = bind(distribution, generator);
    for (int i=0; i<10; ++i)
        cout << roller() << " ";
    cout << endl;


    random_device seeder;
    const auto seed = seeder.entropy() ? seeder() : time(nullptr);
    default_random_engine generator1(
        static_cast<default_random_engine::result_type>(seed));
    uniform_int_distribution<int> distribution1(1, 6);
    int dice_roll1 = distribution1(generator1);
    cout << "Random number: " << dice_roll1 << endl;


    mt19937 generator2;
    vector<double> intervals = {1, 20, 40, 60, 80};
    vector<double> weights = {1, 3, 1, 3};
    piecewise_constant_distribution<double> distribution2(
        begin(intervals), end(intervals), begin(weights));
    int value = static_cast<int>(distribution2(generator2));
    cout << "mystery value: " << value << endl;

    return 0;
}
