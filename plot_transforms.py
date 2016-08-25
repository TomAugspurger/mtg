import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

FIGSIZE = (4, 6)
choices = [mpl.colors.rgb2hex(c) for c in sns.color_palette()[:4]]
N_BLOCKS = 12
STEP = -1 / N_BLOCKS


def plot_factorize():
    starts = np.arange(1, 0, STEP)
    np.random.seed(4)
    colors = np.random.choice(choices, N_BLOCKS)
    factors, _ = pd.factorize(colors)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for start, c in zip(starts, colors):
        ax.add_patch(
            mpl.patches.Rectangle((0, start), .4, STEP, facecolor=c),
        )

    for start, c, f in zip(starts, colors, factors):
        ax.add_patch(
            mpl.patches.Rectangle((.6, start), .399, STEP - .0001, fill=False),
        )
        ax.annotate(f, xy=(.4, start - .05), xytext=(.775, start - .05),
                    verticalalignment='center',
                    color=c,
                    fontsize=32)
    ax.yaxis.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_xticks([.2, .81])
    ax.tick_params(axis='x', labelsize=24)
    ax.set_xticklabels(['Raw', 'Factorized'])

    ax.set_axis_bgcolor('none')
    plt.grid(0)
    plt.tight_layout()
    plt.savefig('factorize.png', dpi=150, transparent=True)
    plt.close()


def plot_get_dummies():
    WIDTH = .15
    starts = np.arange(1, 0, -1 / 12)
    np.random.seed(4)
    colors = np.random.choice(choices, 12)
    factors, _ = pd.factorize(colors)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for start, c in zip(starts, colors):
        ax.add_patch(
            mpl.patches.Rectangle((0, start), WIDTH, STEP, facecolor=c),
        )

    lefts = [.4, .55, .7, .85]
    for i, left in enumerate(lefts):
        for start, c, f in zip(starts, colors, factors):
            if i != f:
                ax.add_patch(
                    mpl.patches.Rectangle((left, start), WIDTH,
                                          STEP, facecolor='#CECECE')
                )
                ax.annotate('0', xy=(left + .045, start - .05),
                            verticalalignment='center',
                            color='#595959', fontsize=32)

            else:
                ax.add_patch(
                    mpl.patches.Rectangle((left, start), WIDTH, STEP,
                                          facecolor=c),
                )
                ax.annotate('1', xy=(left + .045, start - .05),
                            verticalalignment='center',
                            color='white', fontsize=32)

    ax.yaxis.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_xticks([.075, .48, .63, .78, .93])
    ax.set_xticklabels(['Raw', 'D1', 'D2', 'D3', 'D4'])
    ax.tick_params(axis='x', labelsize=24)

    ax.set_axis_bgcolor('none')
    plt.grid(0)
    plt.tight_layout()
    plt.savefig('dummy.png', dpi=150, transparent=True)
    plt.close()


def main():
    plot_factorize()
    plot_get_dummies()

if __name__ == '__main__':
    main()
