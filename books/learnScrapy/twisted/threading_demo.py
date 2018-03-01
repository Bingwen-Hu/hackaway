import threading
from time import sleep


def install_wordpress(customer):
    """code to mimic some heavy work"""
    print(f"Start installation for {customer}")
    sleep(3)
    print(f"All done for {customer}")

def developers_day(customers):
    """using threading ensure every thread dont comflict"""
    lock = threading.Lock()

    def dev_day(id):
        print(f"Goodmoring from developer {id}")

        # let's lock
        lock.acquire()
        while customers:
            customer = customers.pop(0)
            lock.release()

            # do the work
            install_wordpress(customer)
            lock.acquire()
        lock.release()
        print(f"Bye from developer {id}")
    
    devs = [threading.Thread(target=dev_day, args=(i, )) for i in range(5)]
    # start in morning
    [dev.start() for dev in devs]
    # end for evening
    [dev.join() for dev in devs]

developers_day(["Customer %d" % i for i in range(5)])