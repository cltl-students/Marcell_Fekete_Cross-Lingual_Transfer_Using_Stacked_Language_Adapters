"""
Functions inspired by EXPLAINABOARD to get better interpretations of results.
@author: Marcell Fekete
"""

import os
import glob
import nltk
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


languages = {
    "amh": "Amharic", "chm": "Meadow Mari",
    "fra": "French", "hun": "Hungarian",
    "ind": "Indonesian", "isl": "Icelandic",
    "sme": "Northern Sami", "vie": "Vietnamese"
}

plt.style.use("seaborn")

FACTORS = [
    "src_len",  # length of the source sentence
    "trg_len",  # length of the target sentence
    "src+trg_len",  # sum of the lengths of the source and target sentences
    "src-trg_len",  # difference of the lengths of the source and target sentences
    r"src/trg_len",  # ratio of the lengths of the source and target sentences
    "src_subs",  # source subtokens
    "trg_subs",  # target subtokens,
    "src_subs_len",  # count of the source subtokens
    "trg_subs_len",  # count of the target subtokens,
    r"src/trg_subs_len",  # difference of the counts of the source and target subwords
    "src_fert",  # fertility (avg subword per token) of the source sentence
    "trg_fert",  # fertility (avg subword per token) of the target sentence
    r"src/trg_fert",  # ratio of the fertility scores of the source and target sentences
    "src_unk_count",  # count of UNK tokens in source sentence
    "trg_unk_count",  # count of UNK tokens in target sentence
    "src_unk_ratio",  # ratio of UNK token count to total subword count in source sentence
    "trg_unk_ratio",  # ratio of UNK token count to total subword count in source sentence
    "label"  # binary label (0 if the sentences are not matched and 1 if they are)
]

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True)


def calculate_factors(df_path, bucket_count=4, outpath=None):
    """
    It adds new columns to a target dataframe with the new columns calculated.

    :param df_path: path to input dataframe
    :param bucket_count: the number of buckets to place the data in
    :param outpath: output path for where to save the dataframe if not None
    :return: a dataframe with different factors calculated
    """
    df = pd.read_csv(df_path, sep="\t", index_col=0)

    target_column = "Gold"

    df = df.filter(items=["Sentence",  # source sentence
                          target_column,
                          "K_1_correct"  # whether the model identified the correct element
                          ])

    for factor_name in FACTORS:
        if factor_name == FACTORS[0]:
            df[factor_name] = df["Sentence"].apply(lambda x: len(nltk.word_tokenize(x)))
        elif factor_name == FACTORS[1]:
            df[factor_name] = df[target_column].apply(lambda x: len(nltk.word_tokenize(x)))
        elif factor_name == FACTORS[2]:
            df[factor_name] = df["src_len"] + df["trg_len"]
        elif factor_name == FACTORS[3]:
            df[factor_name] = df["src_len"] - df["trg_len"]
        elif factor_name == FACTORS[4]:
            df[factor_name] = round(df["src_len"] / df["trg_len"], 3)
        elif factor_name == FACTORS[5]:
            df[factor_name] = df["Sentence"].apply(lambda x: tokenizer.tokenize(x))
        elif factor_name == FACTORS[6]:
            df[factor_name] = df[target_column].apply(lambda x: tokenizer.tokenize(x))
        elif factor_name == FACTORS[7]:
            df[factor_name] = df["src_subs"].apply(lambda x: len(x))
        elif factor_name == FACTORS[8]:
            df[factor_name] = df["trg_subs"].apply(lambda x: len(x))
        elif factor_name == FACTORS[9]:
            df[factor_name] = df["src_subs_len"] / df["trg_subs_len"]
        elif factor_name == FACTORS[10]:
            df[factor_name] = round(df["src_subs_len"] / df["src_len"], 3)
        elif factor_name == FACTORS[11]:
            df[factor_name] = round(df["trg_subs_len"] / df["trg_len"], 3)
        elif factor_name == FACTORS[12]:
            df[factor_name] = round(df["src_fert"] / df["trg_fert"], 3)
        elif factor_name == FACTORS[13]:
            df[factor_name] = df["src_subs"].apply(lambda x: x.count('[UNK]'))
        elif factor_name == FACTORS[14]:
            df[factor_name] = df["trg_subs"].apply(lambda x: x.count('[UNK]'))
        elif factor_name == FACTORS[15]:
            df[factor_name] = round(df["src_unk_count"] / df["src_subs_len"], 3)
        elif factor_name == FACTORS[16]:
            df[factor_name] = round(df["trg_unk_count"] / df["trg_subs_len"], 3)
        elif factor_name == FACTORS[17]:
            df[factor_name] = df["K_1_correct"].replace(False, 0).replace(True, 1)

    # then arrange factors into buckets
    for factor_name in FACTORS:
        if factor_name not in {"src_subs", "trg_subs", "label"}:
            if factor_name in {"src_len", "trg_len", "src+trg_len", "src-trg_len",
                               "src_subs_len", "trg_subs_len", "src_unk_count", "trg_unk_count"}:
                if outpath:
                    df[f"bucketed_{factor_name}"] = pd.qcut(df[factor_name], q=bucket_count, duplicates="drop",
                                                            precision=0).astype(str)
                else:
                    df[f"bucketed_{factor_name}"] = pd.qcut(df[factor_name], q=bucket_count, duplicates="drop",
                                                            precision=0)
            else:
                if outpath:
                    df[f"bucketed_{factor_name}"] = pd.qcut(df[factor_name], q=bucket_count, duplicates="drop").astype(str)
                else:
                    df[f"bucketed_{factor_name}"] = pd.qcut(df[factor_name], q=bucket_count,
                                                            duplicates="drop")

    df.drop(["src_subs", "trg_subs"], axis=1, inplace=True)
    df.replace("nan", 0, inplace=True)

    if outpath:
        df.to_csv(outpath, sep=";")
    return df


def return_bucket_values(df, factor):
    """
    Given a dataframe and factor,
    it returns a bucketed representations of the factor.

    :param df: input dataframe
    :param factor: factor to bucket the data on
    :return: a bucketed dataframe
    """
    bucketed_dict = defaultdict(dict)

    bucketed_factor = f"bucketed_{factor}"

    df_filtered = df.filter(items=[bucketed_factor, "label"])
    grouped_df = df_filtered.groupby(bucketed_factor)

    for bucket_group, group in grouped_df:
        correct_count = sum(group["label"])
        bucket_size = len(group)
        accuracy = round(correct_count / bucket_size * 100, 2)
        bucketed_dict[bucket_group]["accuracy"] = accuracy
        bucketed_dict[bucket_group]["size"] = bucket_size

    return pd.DataFrame(bucketed_dict, index=None).T


def plot_bucketed_table(df, factor, output_path, columns_to_plot, **kwargs):
    """
    Provided an input dataframe, the function plots the buckets on a barchart.
    Kwargs provided to the plot.

    :param df: input dataframe
    :param factor: factor that was plotted
    :param output_path: where the plot is saved
    :param columns_to_plot: columns to plot
    :param kwargs: parameters passed to the bar plot
    :return:
    """
    ax = df.plot.bar(y=columns_to_plot, rot=0, legend=False, **kwargs)

    # annotate bar chart for the size of buckets
    for p, s in zip(ax.patches, df["size"]):
        ax.annotate(str(int(s)),
                    (p.get_x() + p.get_width() / 2,  # x coordinate
                     p.get_height()),  # y coordinate
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="bottom")

    if output_path:
        output_path = os.path.join(output_path, f"{factor}.pdf")
        plt.savefig(output_path)
    plt.show()


# # output_factors -- if True, output factors in a dataframe
# output_factors = False
results_path = "results/outputs/**"
output_dir = "results/figures/factors/"
glob_pattern = sorted(glob.glob(results_path))

factor_to_extract = "src_fert"
title_draft = "Accuracy scores per source text fertility per experiment"
graph_parameters = {"ylabel": "Accuracy (out of 100)"}


for language_pair_path in glob_pattern:
    language_df = None
    language_pair_name = os.path.basename(language_pair_path)
    en, lg = language_pair_name.split("-")
    print(lg)
    # collect experiments
    experiments = glob.glob(f"{language_pair_path}/**")
    # iterate through experiments and calculate factor values and bucket them
    for exp in experiments:
        exp_name = os.path.basename(exp)
        experiment_output = os.path.join(exp, "output.tsv")
        output_factor = os.path.join(output_dir, factor_to_extract)
        if not os.path.exists(output_factor):
            os.makedirs(output_factor)
        df_to_calculate_with = calculate_factors(experiment_output, outpath=None)

        # select the values for the particular factor
        df_to_represent = return_bucket_values(df_to_calculate_with, factor_to_extract)

        # if a particular experiment does not support bucketing, skip it
        if "size" not in df_to_represent.columns:
            continue

        df_to_represent.rename(columns={"accuracy": exp_name}, inplace=True)

        # merge values between the same language
        if language_df is None:
            language_df = df_to_represent
        else:
            language_df = language_df.combine_first(df_to_represent)

    if language_df is not None:
        ttr = language_df.filter(items=["no-lang", "target-lang", "low-ttr", "high-ttr"])
        mattr = language_df.filter(items=["low-mattr", "high-mattr"])
        ttr.rename(columns={"low-ttr": "low-dttr", "high-ttr": "high-dttr"}, inplace=True)
        mattr.rename(columns={"low-mattr": "low-dmattr", "high-mattr": "high-dmattr"}, inplace=True)

        df = pd.concat([ttr, mattr], axis=1)

        ax = df.plot.bar(rot=0, **graph_parameters)
        for p, s in zip(ax.patches, language_df["size"]):
            ax.annotate(str(int(s)), (p.get_x() + p.get_width() / 2, p.get_height()),
                        xytext=(0, 0), textcoords="offset points", ha="center", va="bottom")

        ticks = range(len(df))

        labels = []

        if factor_to_extract == "src_fert":
            if lg == "amh":
                labels = ["(1.0,1.1)", "(1.2,1.3)", "(1.4,8.0)"]
            elif lg == "chm":
                labels = ["(1.0,2.0)", "(2.1,2.7)", "(2.8,3.5)", "(3.6,15.0)"]
            elif lg == "sme":
                labels = ["(1.0,2.0)", "(2.1,2.7)", "(2.8,4.0)", "(4.1,11.0)"]
        elif factor_to_extract == "src_unk_ratio":
            if lg == "amh":
                labels = ["(0.05,0.67)", "(0.68,0.94)", "(0.95,1.00)"]
            elif lg == "chm":
                labels = ["(0.00,0.07)", "(0.08-1.00)"]

        if labels:
            plt.xticks(ticks=ticks, labels=labels)
        else:
            plt.xticks(ticks=[])

        plt.savefig(os.path.join(output_factor, f"{lg}.pdf"))
        plt.show()
