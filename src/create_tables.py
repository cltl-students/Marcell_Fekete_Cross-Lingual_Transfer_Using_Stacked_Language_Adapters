import os
import glob
import pandas as pd


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
    output_path = os.path.join(results_path.strip('**'), "all_results", f"{metric}.csv")
    concat_df.to_csv(output_path, sep=";")
    print(output_path)
    print(concat_df)


# order in which the experiments should be arranged
desired_order = ["no-lang", "target-lang", "low-ttr", "high-ttr", "low-mattr", "high-mattr"]
