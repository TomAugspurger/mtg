import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

HEIGHT = .1
choices = [mpl.colors.rgb2hex(c) for c in sns.color_palette()[:3]]


def plot_factorize():
    starts = np.arange(0, 1, .1)
    np.random.seed(4)
    colors = np.random.choice(choices, 10)
    factors, _ = pd.factorize(colors)

    fig, ax = plt.subplots(figsize=(4, 8))
    for start, c in zip(starts, colors):
        ax.add_patch(
            mpl.patches.Rectangle((0, start), .4, HEIGHT, facecolor=c),
        )

    for start, c, f in zip(starts, colors, factors):
        ax.add_patch(
            mpl.patches.Rectangle((.6, start), .4, HEIGHT, fill=False),
        )
        ax.annotate(f, xy=(.4, start + .05), xytext=(.8, start + .05),
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
    plt.savefig('factorize.png', dpi=300)
    plt.close()


def plot_get_dummies():
    WIDTH = .2
    starts = np.arange(0, 1, .1)
    np.random.seed(4)
    colors = np.random.choice(choices, 10)
    factors, _ = pd.factorize(colors)

    fig, ax = plt.subplots()
    for start, c in zip(starts, colors):
        ax.add_patch(
            mpl.patches.Rectangle((0, start), .2, HEIGHT, facecolor=c),
        )

    lefts = [.4, .6, .8]
    for i, left in enumerate(lefts):
        for start, c, f in zip(starts, colors, factors):
            if i != f:
                ax.add_patch(
                    mpl.patches.Rectangle((left, start), WIDTH,
                                          HEIGHT, facecolor='#CECECE')
                )
                ax.annotate('0', xy=(left + .09, start + .04),
                            verticalalignment='center',
                            color='#595959', fontsize=32)

            else:
                ax.add_patch(
                    mpl.patches.Rectangle((left, start), WIDTH, HEIGHT,
                                          facecolor=c),
                )
                ax.annotate('1', xy=(left + .09, start + .04),
                            verticalalignment='center',
                            color='white', fontsize=32)

    ax.yaxis.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_xticks([.1, .5, .7, .9])
    ax.set_xticklabels(['Raw', 'D1', 'D2', 'D3'])
    ax.tick_params(axis='x', labelsize=24)

    ax.set_axis_bgcolor('none')
    plt.grid(0)
    plt.tight_layout()
    plt.savefig('dummy.png', dpi=300)
    plt.close()


def main():
    plot_factorize()
    plot_get_dummies()

if __name__ == '__main__':
    main()
