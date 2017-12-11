import numpy as np
import matplotlib.pyplot as plt

def softmax(x1, x2, x3, beta):
    """Compute softmax probabilities for all actions."""
    xs = np.array((x1, x2, x3))
    num = np.exp(xs * beta)
    den = np.exp(xs * beta).sum()
    return num / den

def plot_softmax(x1, x2, x3, beta, ax):
    probs = softmax(x1, x2, x3, beta)
    y_pos = np.arange(3)
    ax.bar(y_pos, probs, align='center')
    ax.set_xticks(y_pos)
    ax.set_xticklabels((r'$P_1$', r'$P_2$', r'$P_3$'), fontsize=14)
    ax.set_title(r'Action probabilities with $\beta=%.1f$' % beta , fontsize=14)

    ax.set_ylim(0, 1.1)

if __name__ == '__main__':
    (x1, x2, x3) = 5, 5.5, 4.5
    betas = (0.5, 2, 4)
    plt.close('all')
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, beta in zip(axs, betas):
        plot_softmax(x1, x2, x3, beta, ax)
    axs[0].set_ylabel('softmax probability', fontsize=14)
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    plt.tight_layout()
    plt.savefig('../figures/softmax.pdf')
    plt.show()
