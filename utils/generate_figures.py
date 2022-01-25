import matplotlib.pyplot as plt
import os
import sys

def generateFigures(x, map_solution, randomized_solutions, randomized_solutions_labels, save_filename,
                     lims=None, legend_loc="best"):
    fig, ax = plt.subplots(figsize=(5,3))
    for sol, lab in zip(randomized_solutions, randomized_solutions_labels):
        ax.plot(x, sol, linewidth=2, label=lab)
    ax.plot(x, map_solution, linewidth=2, label="MAP solution")
    ax.legend(loc=legend_loc)

    if lims is not None:
        if "xlim" in lims:
            plt.xlim(lims["xlim"])
        if "ylim" in lims:
            plt.ylim(lims["ylim"])
            
    fig.tight_layout()
    print(f"Saving figure to {save_filename}")
    checkDirectory(save_filename)
    
    plt.savefig(save_filename, pad_inches=0, dpi=300)
    plt.close(fig)

def checkDirectory(filepath):
    filename = (filepath.split('/'))[-1]
    path = filepath[:-len(filename)]
    if not os.path.isdir(path):
        os.makedirs(path)
