# chapter7 closure and decorator
"""
the first crucial fact about decorators is that they have the power to
replace the decorated function with a different one. The second crucial fact is that they
are executed immediately when a module is loaded.
"""


registry = []

def register(func):
    print("running register(%s)" % func)
    registry.append(func)
    return func


@register
def f1():
    print('running f1()')

@register
def f2():
    print("running f2()")

def f3():
    print("running f3()")

def main():
    print("running main()")
    print("register->", registry)
    f1()
    f2()
    f3()

#if __name__ == '__main__':
#    main()



#==============================================================================
# a more valuable usage
#==============================================================================

promos = []
def promotion(promo_func):
    promos.append(promo_func)
    return promo_func
@promotion
def fidelity(order):
    """5% discount for customers with 1000 or more fidelity points"""
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0

@promotion
def bulk_item(order):
    """10% discount for each LineItem with 20 or more units"""
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .1
    return discount

@promotion
def large_order(order):
    """7% discount for orders with 10 or more distinct items"""
    distinct_items = {item.product for item in order.cart}
    if len(distinct_items) >= 10:
        return order.total() * .07
    return 0


def best_promo(order):
    """Select best discount available
    """
    return max(promo(order) for promo in promos)




