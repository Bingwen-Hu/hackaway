"""similar to twisted_demo but using more utilities 
provided by twisted library"""

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
    print(f"Good morning from twisted developer")

    # get many installation jobs
    work = [inline_install(customer) for customer in customers]
    # join to a jobs list
    join = defer.DeferredList(work)
    # when all the jobs done, stop the engine
    join.addCallback(lambda _: reactor.stop())
    print("Bye from twisted developer!")

twisted_developer_day(["custmoer %d" % i for i in range(15)])

# fire the reactor
reactor.run()