import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


j = 2
V_j = [i / 2**j for i in range(2**j+1)]
print(V_j)

def Phi(x):
    if 0 <= x <= 1:
        return 1
    else:
        return 0
# continues with all the code block before
# we have j and V_j already
# we discard j in the name for fixed j.
def Phi_ifn(i):
    def Phi_i(x):
        val = 2**(j/2) * Phi(2**j * x - i)
        return val
    return Phi_i

F_2 = [Phi_ifn(i) for i in range(2**j)]

def visual_fn(F_j, figsize):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(*figsize)

    xs = np.linspace(0, 1, 50)
    for i, fn in enumerate(F_j):
        ax = plt.subplot(gs[i])
        y = [fn(x) for x in xs]
        if i != 0:
            ax.set_yticklabels([])
        ax.set_aspect('auto')
        plt.plot(xs, y)
    plt.show()

visual_fn(F_2, [1, 4])