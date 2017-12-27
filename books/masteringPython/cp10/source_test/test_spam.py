class Spam(object):
    def __init__(self, count):
        self.count = count

    def __eq__(self, other):
        return self.count == other.count


def test_spam_equal_correct():
    a = Spam(5)
    b = Spam(5)

    assert a == b


def test_spam_equal_broken():
    a = Spam(5)
    b = Spam(10)

    assert a == b
