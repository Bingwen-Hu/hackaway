# every time you can take away 1-3 stones
# show given a number of stones, can you figure out how to solve it?reverse
def take_stones(n):
    if n % 4 == 0:
        return False
    else:
        res = n % 4
        print("第一次拿走： {}个石头".format(res))
        print("在这之后，每次对方拿n个石头，我们就拿4-n个石头")
        return True

# 回文数
def palindrone(string):
    length = len(string)
    iter_ = length // 2
    flag = True
    for i in range(iter_):
        if string[i] != string[length-1-i]:
            flag = False
            break
    return flag

def reverse(number):
    if number < 0:
        number = abs(number)
        while number % 10 == 0: 
            number = number // 10
        number = str(number)
        number = number[::-1]
        number = -int(number)
    if number >=0:
        while number % 10 == 0: 
            number = number // 10
        number = str(number)
        number = number[::-1]
        number = int(number)
    print(number)


def reverse2(number):
    flag = False
    if number < 0:
        flag = True
        number = abs(number)

    while number % 10 == 0: 
        number = number // 10
    number = str(number)
    number = number[::-1]
    number = int(number)

    if flag == True:
        number = -number
    print(number)
        
         
    


