from twisted.internet import reactor
from twisted.internet import defer
from twisted.internet import task

# semms very strange
def schedule_install(customer):
    """define a specific job and bind a callback to it"""
    def schedule_install_wordpress():    
        """a task or a join?"""
        def on_done():
            print(f"callback: finished installation for {customer}")
        print(f"Scheduling: installation for {customer}")
        return task.deferLater(reactor, 3, on_done)

    def all_done(_):
        print(f'All done for {customer}')
    
    d = schedule_install_wordpress() 
    d.addCallback(all_done)
    return d
        
def twisted_developer_day(customers):
    print(f"Good morning from twisted developer")

    # get many installation jobs
    work = [schedule_install(customer) for customer in customers]
    # join to a jobs list
    join = defer.DeferredList(work)
    # when all the jobs done, stop the engine
    join.addCallback(lambda _: reactor.stop())
    print("Bye from twisted developer!")

twisted_developer_day(["custmoer %d" % i for i in range(15)])

# fire the reactor
reactor.run()