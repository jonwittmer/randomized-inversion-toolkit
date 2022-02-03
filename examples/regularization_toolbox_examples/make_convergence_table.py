import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

filename = sys.argv[1]
problem_name = filename.split('_')[0]

data = pd.read_csv(filename, sep=',', index_col=0)
print(data)
samples = list(data)
samples = [int(sample.strip()) for sample in samples]
methods = list(data.index.values)
print(samples)
print(methods)

fig, ax = plt.subplots(figsize=(5,3.5))
for method in methods:
    ax.semilogy(samples, data.loc[[method]].to_numpy().reshape((len(samples),)), label=method)
ax.legend(loc='upper right')
ax.set_ylim(ymax=1e5)
ax.set_ylabel("relative error, %")
ax.set_xlabel("number of samples, N")
plt.tight_layout()
plt.savefig(f"figures/{problem_name}/error_plot.pdf", dpi=300)
#x = data[0,:]

