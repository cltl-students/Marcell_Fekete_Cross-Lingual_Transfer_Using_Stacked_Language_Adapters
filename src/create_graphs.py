"""
Functions that allow representing all the results on graphs.
@author: Marcell Fekete
"""


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")

path = 'results/tables/all_results/**.csv'

for table_path in glob.glob(path):
    metric_name = os.path.basename(table_path).strip(".csv")

    # scores pertaining to overall type-token ratio
    ttr = pd.read_csv(table_path, sep=";", index_col=0).filter(
        items=["no-lang", "target-lang", "low-ttr", "high-ttr"])

    # scores pertaining to moving-average type-token ratio
    mattr = pd.read_csv(table_path, sep=";", index_col=0).filter(
        items=["no-lang", "target-lang", "low-mattr", "high-mattr"])

    # figure per setting
    xlabel = "Target languages"
    if metric_name == "accuracy":
        title = "Accuracy scores (k=1) per language with different experiments."
        ylabel = "Accuracy scores (out of 100)"
    elif metric_name == "k_3_accuracy":
        title = "Accuracy scores (k=3) per language with different experiments."
        ylabel = "Accuracy scores (out of 100)"
    else:
        title = "Mean reciprocal ranks per language with different experiments."
        ylabel = "Mean reciprocal rank (out of 1)"

    # ttr
    ax = ttr.plot.bar(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        rot=0)

    plt.savefig(os.path.join(
        "results/figures/all_results", f"{metric_name}_ttr.pdf"
    ))
    plt.show()

    # mattr
    ax = mattr.plot.bar(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        rot=0)

    plt.savefig(os.path.join(
        "results/figures/all_results", f"{metric_name}_mattr.pdf"
    ))
    plt.show()
