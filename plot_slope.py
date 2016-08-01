import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='ticks', context='talk')

df = pd.read_csv('tips.csv')

ax = sns.stripplot('day', 'tip', data=df, jitter=1)
xlim = ax.get_xlim()
xx = np.linspace(*xlim, 2)
ax.plot(xx, [3.2245 - xx[0] * .2060, 3.2245 - xx[1] * .2060], lw=4)
sns.despine()
plt.savefig("figures/factorize_slope.png", dpi=200)
