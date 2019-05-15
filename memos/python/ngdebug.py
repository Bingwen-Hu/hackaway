import pysnooper

@pysnooper.snoop()
def mytest(x):
    def helper(y):
        return y + x
    return helper


if __name__ == '__main__':
    f = mytest(4)
    result = f(19)

    with pysnooper.snoop():
        mory = "Mory Owl"
        ann = "Ann"
        