import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def visual_mnist(epoch, samples, figsize):
    """visualize mnist data

    Args:
        epoch: integer
        samples: numpy format 
        figsize: figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(*figsize)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    os.makedirs('output/', exist_ok=True)
    plt.savefig(f'output/{str(epoch).zfill(3)}.png', bbox_inches='tight')
    plt.close(fig)