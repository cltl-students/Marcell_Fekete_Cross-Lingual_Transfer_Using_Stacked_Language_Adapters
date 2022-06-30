"""
Functions that aggregate results across languages.
@author: Marcell Fekete
"""


import os
import json
import glob
import pandas as pd
from collections import defaultdict


# output files for the experiments
results_path = 'results/outputs/**'
language_pairs = sorted(glob.glob(results_path))


def merge_scores(metric="accuracy"):
    """
    Merges scores into a single dataframe,
    one row per language.

    @help from Shahrin Rahman Maimuna

    :param metric: 'accuracy', 'k_3_accuracy', or 'mean reciprocal rank'
    :return: None
    """
    merged_dfs = []

    for language_pair in language_pairs:
        en, trg = os.path.basename(language_pair).split('-')
        score_path = os.path.join(language_pair, f"{trg}_scores.csv")
        df = pd.read_csv(score_path, sep=";", index_col=0)
        accuracy = pd.DataFrame(df.loc[metric]).T
        accuracy = accuracy.rename(index={metric: trg})
        merged_dfs.append(accuracy)

    concat_df = pd.concat(merged_dfs)
    output_path = os.path.join("results/tables/all_results/", f"{metric}.csv")
    concat_df.to_csv(output_path, sep=";")
    print(output_path)
    print(concat_df)


# order in which the experiments should be arranged in the table
desired_order = ["no-lang", "target-lang", "low-ttr", "high-ttr", "low-mattr", "high-mattr"]

# looping through all language pairs
for language_pair in language_pairs:
    output_scores = defaultdict(dict)
    # returning target language name
    en, trg = os.path.basename(language_pair).split('-')
    language_pair_folder = os.path.join(language_pair, "**")
    # looping through experiments
    for experiment_folder in sorted(glob.glob(language_pair_folder)):
        experiment = os.path.basename(experiment_folder)
        score_path = os.path.join(experiment_folder, "scores.txt")
        # returning scores from the experiment
        with open(score_path, 'r') as infile:
            score_dict = dict()
            scores = infile.read().replace("'", "\"").split("\n")
            scores = [json.loads(score) for score in scores]
            for score in scores:
                for k, v in score.items():
                    score_dict[k] = round(v, 3)

            output_scores[experiment] = score_dict

    df = pd.DataFrame.from_dict(output_scores)
    df = df[desired_order]
    output_path = os.path.join("results/tables/per_language/", f"{trg}_scores.csv")

    df.to_csv(output_path, sep=";")

    print(df)

merge_scores("accuracy")
merge_scores("k_3_accuracy")
merge_scores("mean reciprocal rank")
