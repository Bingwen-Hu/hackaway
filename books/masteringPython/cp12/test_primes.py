# using line_profiler to test
# kernprof -l test_primes.py
# python3 -m line_profiler test_primes.py.lprof

import itertools


@profile  # come from line_profiler
def primes():
    n = 2
    primes = set()
    while True:
        for p in primes:
            if n % p == 0:
                break
        else:
            primes.add(n)
            yield n
        n += 1


if __name__ == '__main__':
    total = 0
    n = 1000
    for prime in itertools.islice(primes(), n):
        total += prime
    print('The sum of first %d primes is %d' % (n, total))
