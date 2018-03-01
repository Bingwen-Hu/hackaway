""" limit the resource the program may consume """
from twisted.internet import reactor
from twisted.internet import defer
from twisted.internet import task

@defer.inlineCallbacks
def inline_install(customer):
    """in this example, code can be read sequently
    and callback function here is not needed"""
    print(f"Scheduling: installation for {customer}")
    yield task.deferLater(reactor, 3, lambda: None)
    print(f"callback: finished installation for {customer}")
    print(f'All done for {customer}')


def twisted_developer_day(customers):
    print("Cood morning from twisted developer")

    work = (inline_install(customer) for customer in customers)
    coop = task.Cooperator()
    # every time get 5 jobs as a batch
    join = defer.DeferredList([coop.coiterate(work) for i in range(5)])
    join.addCallback(lambda _: reactor.stop())
    print("Bye from Twisted developer!")


twisted_developer_day(["custmoer %d" % i for i in range(15)])

# fire the reactor
reactor.run()