# I have $100, change to $10 and $5 and $2 and $1
# how many kinds of change can I do?

from pprint import pprint

def change(amount, coins):
    maxes = [amount // coin +1 for coin in coins]
    count = 0
    ways = []
    for c1 in range(maxes[0]):
        for c2 in range(maxes[1]):
            for c3 in range(maxes[2]):
                for c4 in range(maxes[3]):
                    if c1 * coins[0] + c2 * coins[1] + c3 * coins[2] + c4 * coins[3] == amount:
                        count += 1 
                        ways.append(dict(zip(coins, [c1,c2,c3,c4])))
    return count, ways

if __name__ == '__main__':
    amount = 7
    coins = [1, 2, 5, 10]
    num, ways = change(amount, coins)
    print(f"Change ${amount} into dollars of {coins} " 
        f"there are {num} ways")
    for way in ways[:10]:
        pprint(way)