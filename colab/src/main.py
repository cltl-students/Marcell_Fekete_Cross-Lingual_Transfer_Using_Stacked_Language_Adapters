"""
Main script to carry out the experiments.
@author: Marcell Fekete
"""


import os
import types
import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
from statistics import mean
from configparser import ConfigParser
from utils.experiments import (get_scores_predictions, accuracy, load_config_list,
                               load_lang_adapters, random_baseline, k_3_accuracy,
                               reciprocal_rank)
from transformers import (AutoTokenizer, AutoConfig, AutoModelWithHeads, pipeline)


def average_pooling(sentence):
    """
    Given an input sentence it retrieves the average of the token embeddings.

    :param sentence: str
    :return: numpy array of embeddings
    """
    embeddings = []
    embeds = pipe(sentence, framework="pt", device=0)[0]
    for embed in embeds:
        embed = np.asarray(embed).astype(np.float32)
        embeddings.append(embed)
    return np.mean(embeddings, axis=0)


def process_sentences(sentences):
    """
    Create sentence embeddings from the mean of the pooled subword tokens
    for multiple sentences.

    :param sentences:
    :return: numpy array of average pooled embeddings
    """
    pooled_sentences = []
    with tqdm(total=len(sentences), desc="Iterating: ") as pbar:
        for sentence in sentences:
            pooled_sentence = average_pooling(sentence)
            pooled_sentences.append(pooled_sentence)
            pbar.update(1)
    return np.asarray(pooled_sentences)


# function from https://stackoverflow.com/questions/71240331/getting-an-error-even-after-using-truncation-for-tokenizer-while-predicting-mlm
def _my_preprocess(self, inputs, return_tensors=None, **preprocess_parameters):
    if return_tensors is None:
        return_tensors = self.framework
    model_inputs = self.tokenizer(inputs, truncation=True, max_length=512,
                                  return_tensors=return_tensors)
    return model_inputs


def main(path_to_source=None,
         path_to_target=None,
         path_to_labels=None,
         source_adapters=None,
         target_adapters=None,
         save_source_path=None,
         general_output=None,
         experiment=None,
         model=None,
         k=3):
    """
    Function to call a single Tatoeba translation pair detection experiment.

    :param path_to_source: source file containing English sentences
    :param path_to_target: target file containing target language sentences
    :param path_to_labels: filepath to labels
    :param source_adapters: list of source adapters
    :param target_adapters: list of target adapters
    :param save_source_path: if not None, place to save source embeddings
    :param general_output: general output directory
    :param experiment: folder for experiment
    :param model: language model
    :param k: number of hits to return
    :return:
    """
    # load source sentences
    with open(path_to_source, 'r') as infile:
        src_set = infile.read().split('\n')[:-1]

    # load target sentences
    with open(path_to_target, 'r') as infile:
        trg_set = infile.read().split('\n')[:-1]

    # load labels and convert them into numerical values
    with open(path_to_labels, 'r') as infile:
        labels = infile.read().split('\n')[:-1]
    labels = [int(label) for label in labels]

    # load source language adapters
    if source_adapters[0] != 'None':
        load_lang_adapters(source_adapters, model)

    # if save_source_path is provided and it doesn't exist,
    # source embeddings are saved in target folder
    if save_source_path and source_adapters[0] != 'None':
        if not os.path.exists(save_source_path):
            os.makedirs(save_source_path)

            src_embeddings = process_sentences(src_set)
            joblib.dump(src_embeddings, os.path.join(save_source_path, 'embs.pkl'))

        else:
            # if save_source_path is provided and it exists,
            # load source embeddings from target folder
            src_embeddings = joblib.load(os.path.join(save_source_path, 'embs.pkl'))

    else:
        src_embeddings = process_sentences(src_set)

    # load target language adapters
    if target_adapters[0] != 'None':
        load_lang_adapters(target_adapters, model)

    # process target embeddings
    trg_embeddings = process_sentences(trg_set)

    # return candidate translations
    scores, predictions = get_scores_predictions(src_embeddings, trg_embeddings, k=int(k))

    # k = 1 evaluation
    top_k = [p[0] for p in predictions]
    top_scores = [s[0] for s in scores]
    top_k_metric = accuracy(labels, top_k)
    top_correct = [str(p == l) for p, l in zip(top_k, labels)]

    # k = 3 evaluation
    k_3 = [p.tolist() for p in predictions]
    k_3_scores = [s.tolist() for s in scores]
    k_3_metric = k_3_accuracy(labels, k_3)
    k_3_correct = [str(l in p) for p, l in zip(k_3, labels)]

    # reciprocal ranks
    reciprocal_ranks = [reciprocal_rank(l, p) for p, l in zip(k_3, labels)]
    mean_reciprocal_rank = {"mean reciprocal rank": round(mean(reciprocal_ranks), 3)}

    # return sentences
    predicted_sentences = [src_set[label] for label in top_k]
    true_match = [src_set[label] for label in labels]

    output_dataframe = pd.DataFrame({
        "Sentence": trg_set,
        "Predicted": predicted_sentences,
        "Gold": true_match,
        "K_1_correct": top_correct,
        "K_3_correct": k_3_correct,
        "K_1_scores": top_scores,
        "K_1_predicted_indices": top_k,
        "K_3_predicted_indices": [', '.join(str(p_item) for p_item in p) for p in k_3],
        "Reciprocal_ranks": [str(rank) for rank in reciprocal_ranks],
        "True_indices": labels,
    })

    # output path for correct experiment
    if not os.path.exists(general_output):
        os.makedirs(general_output)

    output_path = os.path.join(general_output, experiment)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # output metrics
    metric_path = os.path.join(output_path, "scores.txt")
    with open(metric_path, "w") as outfile:
        outfile.write(f'{top_k_metric}\n')
        outfile.write(f'{k_3_metric}\n')
        outfile.write(f'{mean_reciprocal_rank}')

    # output dataframe
    dataframe_path = os.path.join(output_path, "output.tsv")

    output_dataframe.to_csv(dataframe_path, sep="\t")

    print()
    print(top_k_metric)
    print(k_3_metric)
    print(mean_reciprocal_rank)
    print()


if __name__ == "__main__":

    cfg = ConfigParser()
    cfg.read_file(open("/content/gdrive/MyDrive/Marcell_Fekete_Cross-Lingual_Transfer_Using_Stacked_Language_Adapters/config.ini"))
    cfg = cfg['DEFAULT']

    model_name = "bert-base-multilingual-cased"

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithHeads.from_pretrained(model_name,
                                               config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(task="feature-extraction",
                    model=model,
                    tokenizer=tokenizer)

    # preprocess method is replaced to allow the truncation of input
    pipe.preprocess = types.MethodType(_my_preprocess, pipe)

    main_parameters = {
        "path_to_source": cfg["path_to_source"],
        "path_to_target": cfg["path_to_target"],
        "path_to_labels": cfg["path_to_labels"],
        "save_source_path": cfg["save_source_path"],
        "general_output": cfg["output_directory"],
        "model": model,
        "k": 3
    }

    # if config is multiconfig, multiple experiments are ran from the same file
    if cfg['multiconfig'] == 'True':

        experiments = load_config_list(cfg['experiments'])
        source_adapters = load_config_list(cfg['source_adapters'],
                                           multiconfig=True)
        target_adapters = load_config_list(cfg['target_adapters'],
                                           multiconfig=True)

        # make sure these lists are the same length
        if len(experiments) != len(source_adapters):
            raise ValueError("Number of output directories doesn't match the number of adapter setups.")
        if len(source_adapters) != len(target_adapters):
            raise ValueError("Number of source and target adapter setups doesn't match up.")

        for exp, source_adps, target_adps in zip(experiments, source_adapters, target_adapters):
            custom_parameters = {"experiment": exp,
                                 "source_adapters": source_adps,
                                 "target_adapters": target_adps}
            main(**main_parameters,
                 **custom_parameters)

    else:
        experiments = cfg["experiments"]
        source_adapters = load_config_list(cfg["source_adapters"])
        target_adapters = load_config_list(cfg["target_adapters"])

        custom_parameters = {"experiment": experiments,
                             "source_adapters": source_adapters,
                             "target_adapters": target_adapters}

        main(**main_parameters, **custom_parameters)
