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

def change_dynamic(amount, coins):
    if len(coins) and amount:
        coin = coins[0]
        many = amount // coin
        with_coin = [change_dynamic(amount - i*coin, coins[1:]) for i in range(1, many+1)]
        without_coin = change_dynamic(amount, coins[1:])
        return sum(with_coin) + without_coin
    elif amount == 0:
        return 1
    else:
        return 0
    
 
def change_less_coins_easy(amount, num=0):
    """change amount by coins but use less nubmer of coins"""
    if amount == 0:
        return num
    if amount < 0:
        return 999
    # coins = [1, 2, 5]
    print("call!")
    number = min([
        change_less_coins_easy(amount-1, num+1),
        change_less_coins_easy(amount-2, num+1),
        change_less_coins_easy(amount-5, num+1),
    ])
    return number
  
# how to reduce useless search?
# NOTE: perhaps use  of all coins
def change_less_coins(amount, coins, num=0):
    """change amount by coins but use less nubmer of coins"""
    if amount == 0:
        return num
    number = min([change_less_coins(amount - coin, coins, num+1) for coin in coins if amount - coin >= 0])
    return number
  


if __name__ == '__main__':
    amount = 21
    coins = [1, 2, 5, 10]
    num, ways = change(amount, coins)
    print(f"Change ${amount} into dollars of {coins} " 
        f"there are {num} ways")
    for way in ways[:10]:
        pprint(way)
    
    way = change_dynamic(amount, coins)
    number = change_less_coins(100, coins)
    print(number)