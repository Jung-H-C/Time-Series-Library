import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

T = 100
t = np.arange(T)

signal = (
    0.25*np.sin(2*np.pi*t/18) +
    0.15*np.sin(2*np.pi*t/7) +
    1.8*np.exp(-((t-55)/12)**2) +
    0.6*np.exp(-((t-25)/6)**2) +
    0.1*np.random.randn(T)
)

abs_delta = np.abs(signal[1:] - signal[:-1])
tb = np.arange(1, T)

# quantiles
q25 = np.quantile(abs_delta, 0.25)
q75 = np.quantile(abs_delta, 0.75)

top_idx = abs_delta >= q75
bot_idx = abs_delta <= q25
mid_idx = (~top_idx) & (~bot_idx)
base_bar_color = "#b7dcff"

plt.figure(figsize=(10,4), dpi=300)

# orange line
plt.plot(t, signal, linewidth=2, color='orange')

# middle bins (faded)
plt.bar(
    tb[mid_idx],
    abs_delta[mid_idx],
    width=0.8,
    alpha=0.25,
    color=base_bar_color,
)

# bottom q% bins
plt.bar(
    tb[bot_idx],
    abs_delta[bot_idx],
    width=0.8,
    alpha=0.8,
    color=base_bar_color,
)

# top q% bins
plt.bar(
    tb[top_idx],
    abs_delta[top_idx],
    width=0.8,
    alpha=0.9,
    color=base_bar_color,
)

ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

path = "./orange.png"
plt.savefig(path)

path
